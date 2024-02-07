class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 7, 7]", primals_2: "f32[64]", primals_3: "f32[64]", primals_4: "f32[64, 64, 1, 1]", primals_5: "f32[64]", primals_6: "f32[64]", primals_7: "f32[64, 64, 3, 3]", primals_8: "f32[64]", primals_9: "f32[64]", primals_10: "f32[256, 64, 1, 1]", primals_11: "f32[256]", primals_12: "f32[256]", primals_13: "f32[256, 64, 1, 1]", primals_14: "f32[256]", primals_15: "f32[256]", primals_16: "f32[64, 256, 1, 1]", primals_17: "f32[64]", primals_18: "f32[64]", primals_19: "f32[64, 64, 3, 3]", primals_20: "f32[64]", primals_21: "f32[64]", primals_22: "f32[256, 64, 1, 1]", primals_23: "f32[256]", primals_24: "f32[256]", primals_25: "f32[64, 256, 1, 1]", primals_26: "f32[64]", primals_27: "f32[64]", primals_28: "f32[64, 64, 3, 3]", primals_29: "f32[64]", primals_30: "f32[64]", primals_31: "f32[256, 64, 1, 1]", primals_32: "f32[256]", primals_33: "f32[256]", primals_34: "f32[128, 256, 1, 1]", primals_35: "f32[128]", primals_36: "f32[128]", primals_37: "f32[128, 128, 3, 3]", primals_38: "f32[128]", primals_39: "f32[128]", primals_40: "f32[512, 128, 1, 1]", primals_41: "f32[512]", primals_42: "f32[512]", primals_43: "f32[512, 256, 1, 1]", primals_44: "f32[512]", primals_45: "f32[512]", primals_46: "f32[128, 512, 1, 1]", primals_47: "f32[128]", primals_48: "f32[128]", primals_49: "f32[128, 128, 3, 3]", primals_50: "f32[128]", primals_51: "f32[128]", primals_52: "f32[512, 128, 1, 1]", primals_53: "f32[512]", primals_54: "f32[512]", primals_55: "f32[128, 512, 1, 1]", primals_56: "f32[128]", primals_57: "f32[128]", primals_58: "f32[128, 128, 3, 3]", primals_59: "f32[128]", primals_60: "f32[128]", primals_61: "f32[512, 128, 1, 1]", primals_62: "f32[512]", primals_63: "f32[512]", primals_64: "f32[128, 512, 1, 1]", primals_65: "f32[128]", primals_66: "f32[128]", primals_67: "f32[128, 128, 3, 3]", primals_68: "f32[128]", primals_69: "f32[128]", primals_70: "f32[512, 128, 1, 1]", primals_71: "f32[512]", primals_72: "f32[512]", primals_73: "f32[256, 512, 1, 1]", primals_74: "f32[256]", primals_75: "f32[256]", primals_76: "f32[256, 256, 3, 3]", primals_77: "f32[256]", primals_78: "f32[256]", primals_79: "f32[1024, 256, 1, 1]", primals_80: "f32[1024]", primals_81: "f32[1024]", primals_82: "f32[1024, 512, 1, 1]", primals_83: "f32[1024]", primals_84: "f32[1024]", primals_85: "f32[256, 1024, 1, 1]", primals_86: "f32[256]", primals_87: "f32[256]", primals_88: "f32[256, 256, 3, 3]", primals_89: "f32[256]", primals_90: "f32[256]", primals_91: "f32[1024, 256, 1, 1]", primals_92: "f32[1024]", primals_93: "f32[1024]", primals_94: "f32[256, 1024, 1, 1]", primals_95: "f32[256]", primals_96: "f32[256]", primals_97: "f32[256, 256, 3, 3]", primals_98: "f32[256]", primals_99: "f32[256]", primals_100: "f32[1024, 256, 1, 1]", primals_101: "f32[1024]", primals_102: "f32[1024]", primals_103: "f32[256, 1024, 1, 1]", primals_104: "f32[256]", primals_105: "f32[256]", primals_106: "f32[256, 256, 3, 3]", primals_107: "f32[256]", primals_108: "f32[256]", primals_109: "f32[1024, 256, 1, 1]", primals_110: "f32[1024]", primals_111: "f32[1024]", primals_112: "f32[256, 1024, 1, 1]", primals_113: "f32[256]", primals_114: "f32[256]", primals_115: "f32[256, 256, 3, 3]", primals_116: "f32[256]", primals_117: "f32[256]", primals_118: "f32[1024, 256, 1, 1]", primals_119: "f32[1024]", primals_120: "f32[1024]", primals_121: "f32[256, 1024, 1, 1]", primals_122: "f32[256]", primals_123: "f32[256]", primals_124: "f32[256, 256, 3, 3]", primals_125: "f32[256]", primals_126: "f32[256]", primals_127: "f32[1024, 256, 1, 1]", primals_128: "f32[1024]", primals_129: "f32[1024]", primals_130: "f32[512, 1024, 1, 1]", primals_131: "f32[512]", primals_132: "f32[512]", primals_133: "f32[512, 512, 3, 3]", primals_134: "f32[512]", primals_135: "f32[512]", primals_136: "f32[2048, 512, 1, 1]", primals_137: "f32[2048]", primals_138: "f32[2048]", primals_139: "f32[2048, 1024, 1, 1]", primals_140: "f32[2048]", primals_141: "f32[2048]", primals_142: "f32[512, 2048, 1, 1]", primals_143: "f32[512]", primals_144: "f32[512]", primals_145: "f32[512, 512, 3, 3]", primals_146: "f32[512]", primals_147: "f32[512]", primals_148: "f32[2048, 512, 1, 1]", primals_149: "f32[2048]", primals_150: "f32[2048]", primals_151: "f32[512, 2048, 1, 1]", primals_152: "f32[512]", primals_153: "f32[512]", primals_154: "f32[512, 512, 3, 3]", primals_155: "f32[512]", primals_156: "f32[512]", primals_157: "f32[2048, 512, 1, 1]", primals_158: "f32[2048]", primals_159: "f32[2048]", primals_160: "f32[1000, 2048]", primals_161: "f32[1000]", primals_162: "f32[64]", primals_163: "f32[64]", primals_164: "i64[]", primals_165: "f32[64]", primals_166: "f32[64]", primals_167: "i64[]", primals_168: "f32[64]", primals_169: "f32[64]", primals_170: "i64[]", primals_171: "f32[256]", primals_172: "f32[256]", primals_173: "i64[]", primals_174: "f32[256]", primals_175: "f32[256]", primals_176: "i64[]", primals_177: "f32[64]", primals_178: "f32[64]", primals_179: "i64[]", primals_180: "f32[64]", primals_181: "f32[64]", primals_182: "i64[]", primals_183: "f32[256]", primals_184: "f32[256]", primals_185: "i64[]", primals_186: "f32[64]", primals_187: "f32[64]", primals_188: "i64[]", primals_189: "f32[64]", primals_190: "f32[64]", primals_191: "i64[]", primals_192: "f32[256]", primals_193: "f32[256]", primals_194: "i64[]", primals_195: "f32[128]", primals_196: "f32[128]", primals_197: "i64[]", primals_198: "f32[128]", primals_199: "f32[128]", primals_200: "i64[]", primals_201: "f32[512]", primals_202: "f32[512]", primals_203: "i64[]", primals_204: "f32[512]", primals_205: "f32[512]", primals_206: "i64[]", primals_207: "f32[128]", primals_208: "f32[128]", primals_209: "i64[]", primals_210: "f32[128]", primals_211: "f32[128]", primals_212: "i64[]", primals_213: "f32[512]", primals_214: "f32[512]", primals_215: "i64[]", primals_216: "f32[128]", primals_217: "f32[128]", primals_218: "i64[]", primals_219: "f32[128]", primals_220: "f32[128]", primals_221: "i64[]", primals_222: "f32[512]", primals_223: "f32[512]", primals_224: "i64[]", primals_225: "f32[128]", primals_226: "f32[128]", primals_227: "i64[]", primals_228: "f32[128]", primals_229: "f32[128]", primals_230: "i64[]", primals_231: "f32[512]", primals_232: "f32[512]", primals_233: "i64[]", primals_234: "f32[256]", primals_235: "f32[256]", primals_236: "i64[]", primals_237: "f32[256]", primals_238: "f32[256]", primals_239: "i64[]", primals_240: "f32[1024]", primals_241: "f32[1024]", primals_242: "i64[]", primals_243: "f32[1024]", primals_244: "f32[1024]", primals_245: "i64[]", primals_246: "f32[256]", primals_247: "f32[256]", primals_248: "i64[]", primals_249: "f32[256]", primals_250: "f32[256]", primals_251: "i64[]", primals_252: "f32[1024]", primals_253: "f32[1024]", primals_254: "i64[]", primals_255: "f32[256]", primals_256: "f32[256]", primals_257: "i64[]", primals_258: "f32[256]", primals_259: "f32[256]", primals_260: "i64[]", primals_261: "f32[1024]", primals_262: "f32[1024]", primals_263: "i64[]", primals_264: "f32[256]", primals_265: "f32[256]", primals_266: "i64[]", primals_267: "f32[256]", primals_268: "f32[256]", primals_269: "i64[]", primals_270: "f32[1024]", primals_271: "f32[1024]", primals_272: "i64[]", primals_273: "f32[256]", primals_274: "f32[256]", primals_275: "i64[]", primals_276: "f32[256]", primals_277: "f32[256]", primals_278: "i64[]", primals_279: "f32[1024]", primals_280: "f32[1024]", primals_281: "i64[]", primals_282: "f32[256]", primals_283: "f32[256]", primals_284: "i64[]", primals_285: "f32[256]", primals_286: "f32[256]", primals_287: "i64[]", primals_288: "f32[1024]", primals_289: "f32[1024]", primals_290: "i64[]", primals_291: "f32[512]", primals_292: "f32[512]", primals_293: "i64[]", primals_294: "f32[512]", primals_295: "f32[512]", primals_296: "i64[]", primals_297: "f32[2048]", primals_298: "f32[2048]", primals_299: "i64[]", primals_300: "f32[2048]", primals_301: "f32[2048]", primals_302: "i64[]", primals_303: "f32[512]", primals_304: "f32[512]", primals_305: "i64[]", primals_306: "f32[512]", primals_307: "f32[512]", primals_308: "i64[]", primals_309: "f32[2048]", primals_310: "f32[2048]", primals_311: "i64[]", primals_312: "f32[512]", primals_313: "f32[512]", primals_314: "i64[]", primals_315: "f32[512]", primals_316: "f32[512]", primals_317: "i64[]", primals_318: "f32[2048]", primals_319: "f32[2048]", primals_320: "i64[]", primals_321: "f32[64, 3, 224, 224]"):
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:305, code: x = self.conv1(x)
        convolution: "f32[64, 64, 112, 112]" = torch.ops.aten.convolution.default(primals_321, primals_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:307, code: x = self.bn1(x)
        add: "i64[]" = torch.ops.aten.add.Tensor(primals_164, 1)
        var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 64, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 64, 1, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[64, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[64, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_1: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_1: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
        mul_2: "f32[64]" = torch.ops.aten.mul.Tensor(primals_162, 0.9)
        add_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000012456169853);  squeeze_2 = None
        mul_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[64]" = torch.ops.aten.mul.Tensor(primals_163, 0.9)
        add_3: "f32[64]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1)
        unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[64, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
        unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[64, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:308, code: x = self.relu(x)
        relu: "f32[64, 64, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:309, code: x = self.maxpool(x)
        max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2], [1, 1])
        getitem_2: "f32[64, 64, 56, 56]" = max_pool2d_with_indices[0]
        getitem_3: "i64[64, 64, 56, 56]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:151, code: out = self.conv1(x)
        convolution_1: "f32[64, 64, 56, 56]" = torch.ops.aten.convolution.default(getitem_2, primals_4, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_167, 1)
        var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[1, 64, 1, 1]" = var_mean_1[0]
        getitem_5: "f32[1, 64, 1, 1]" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_1: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1: "f32[64, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_5)
        mul_7: "f32[64, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_4: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_8: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
        mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(primals_165, 0.9)
        add_7: "f32[64]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000049824865598);  squeeze_5 = None
        mul_11: "f32[64]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(primals_166, 0.9)
        add_8: "f32[64]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
        unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[64, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
        unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_9: "f32[64, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:153, code: out = self.relu(out)
        relu_1: "f32[64, 64, 56, 56]" = torch.ops.aten.relu.default(add_9);  add_9 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:155, code: out = self.conv2(out)
        convolution_2: "f32[64, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_1, primals_7, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_170, 1)
        var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem_6: "f32[1, 64, 1, 1]" = var_mean_2[0]
        getitem_7: "f32[1, 64, 1, 1]" = var_mean_2[1];  var_mean_2 = None
        add_11: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_2: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_2: "f32[64, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_7)
        mul_14: "f32[64, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_7: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_15: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
        mul_16: "f32[64]" = torch.ops.aten.mul.Tensor(primals_168, 0.9)
        add_12: "f32[64]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_17: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000049824865598);  squeeze_8 = None
        mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(primals_169, 0.9)
        add_13: "f32[64]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1)
        unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[64, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1);  primals_9 = None
        unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_14: "f32[64, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:157, code: out = self.relu(out)
        relu_2: "f32[64, 64, 56, 56]" = torch.ops.aten.relu.default(add_14);  add_14 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:159, code: out = self.conv3(out)
        convolution_3: "f32[64, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_2, primals_10, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_173, 1)
        var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
        getitem_8: "f32[1, 256, 1, 1]" = var_mean_3[0]
        getitem_9: "f32[1, 256, 1, 1]" = var_mean_3[1];  var_mean_3 = None
        add_16: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_3: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_3: "f32[64, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_9)
        mul_21: "f32[64, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_10: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_22: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
        mul_23: "f32[256]" = torch.ops.aten.mul.Tensor(primals_171, 0.9)
        add_17: "f32[256]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        squeeze_11: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_24: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000049824865598);  squeeze_11 = None
        mul_25: "f32[256]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26: "f32[256]" = torch.ops.aten.mul.Tensor(primals_172, 0.9)
        add_18: "f32[256]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        unsqueeze_12: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
        unsqueeze_13: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_27: "f32[64, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        unsqueeze_14: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
        unsqueeze_15: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_19: "f32[64, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:163, code: residual = self.downsample(x)
        convolution_4: "f32[64, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem_2, primals_13, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_176, 1)
        var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
        getitem_10: "f32[1, 256, 1, 1]" = var_mean_4[0]
        getitem_11: "f32[1, 256, 1, 1]" = var_mean_4[1];  var_mean_4 = None
        add_21: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
        rsqrt_4: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_4: "f32[64, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_11)
        mul_28: "f32[64, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_13: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_29: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
        mul_30: "f32[256]" = torch.ops.aten.mul.Tensor(primals_174, 0.9)
        add_22: "f32[256]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
        squeeze_14: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_31: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000049824865598);  squeeze_14 = None
        mul_32: "f32[256]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
        mul_33: "f32[256]" = torch.ops.aten.mul.Tensor(primals_175, 0.9)
        add_23: "f32[256]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        unsqueeze_16: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1)
        unsqueeze_17: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_34: "f32[64, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        unsqueeze_18: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1);  primals_15 = None
        unsqueeze_19: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_24: "f32[64, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:166, code: out += residual
        add_25: "f32[64, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_19, add_24);  add_19 = add_24 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:173, code: out = self.relu(out)
        relu_3: "f32[64, 256, 56, 56]" = torch.ops.aten.relu.default(add_25);  add_25 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:151, code: out = self.conv1(x)
        convolution_5: "f32[64, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_3, primals_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        add_26: "i64[]" = torch.ops.aten.add.Tensor(primals_179, 1)
        var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
        getitem_12: "f32[1, 64, 1, 1]" = var_mean_5[0]
        getitem_13: "f32[1, 64, 1, 1]" = var_mean_5[1];  var_mean_5 = None
        add_27: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_5: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_5: "f32[64, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_13)
        mul_35: "f32[64, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_16: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_36: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
        mul_37: "f32[64]" = torch.ops.aten.mul.Tensor(primals_177, 0.9)
        add_28: "f32[64]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        squeeze_17: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_38: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000049824865598);  squeeze_17 = None
        mul_39: "f32[64]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
        mul_40: "f32[64]" = torch.ops.aten.mul.Tensor(primals_178, 0.9)
        add_29: "f32[64]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        unsqueeze_20: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
        unsqueeze_21: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41: "f32[64, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
        unsqueeze_23: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_30: "f32[64, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:153, code: out = self.relu(out)
        relu_4: "f32[64, 64, 56, 56]" = torch.ops.aten.relu.default(add_30);  add_30 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:155, code: out = self.conv2(out)
        convolution_6: "f32[64, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_19, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        add_31: "i64[]" = torch.ops.aten.add.Tensor(primals_182, 1)
        var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
        getitem_14: "f32[1, 64, 1, 1]" = var_mean_6[0]
        getitem_15: "f32[1, 64, 1, 1]" = var_mean_6[1];  var_mean_6 = None
        add_32: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
        rsqrt_6: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_6: "f32[64, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_15)
        mul_42: "f32[64, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_19: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_43: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
        mul_44: "f32[64]" = torch.ops.aten.mul.Tensor(primals_180, 0.9)
        add_33: "f32[64]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        squeeze_20: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_45: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000049824865598);  squeeze_20 = None
        mul_46: "f32[64]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
        mul_47: "f32[64]" = torch.ops.aten.mul.Tensor(primals_181, 0.9)
        add_34: "f32[64]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        unsqueeze_24: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1)
        unsqueeze_25: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_48: "f32[64, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
        unsqueeze_26: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1);  primals_21 = None
        unsqueeze_27: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_35: "f32[64, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:157, code: out = self.relu(out)
        relu_5: "f32[64, 64, 56, 56]" = torch.ops.aten.relu.default(add_35);  add_35 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:159, code: out = self.conv3(out)
        convolution_7: "f32[64, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_5, primals_22, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_185, 1)
        var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
        getitem_16: "f32[1, 256, 1, 1]" = var_mean_7[0]
        getitem_17: "f32[1, 256, 1, 1]" = var_mean_7[1];  var_mean_7 = None
        add_37: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
        rsqrt_7: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_7: "f32[64, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_17)
        mul_49: "f32[64, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_22: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_50: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
        mul_51: "f32[256]" = torch.ops.aten.mul.Tensor(primals_183, 0.9)
        add_38: "f32[256]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        squeeze_23: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_52: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000049824865598);  squeeze_23 = None
        mul_53: "f32[256]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
        mul_54: "f32[256]" = torch.ops.aten.mul.Tensor(primals_184, 0.9)
        add_39: "f32[256]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
        unsqueeze_28: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
        unsqueeze_29: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_55: "f32[64, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        unsqueeze_30: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
        unsqueeze_31: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_40: "f32[64, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:166, code: out += residual
        add_41: "f32[64, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_40, relu_3);  add_40 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:173, code: out = self.relu(out)
        relu_6: "f32[64, 256, 56, 56]" = torch.ops.aten.relu.default(add_41);  add_41 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:151, code: out = self.conv1(x)
        convolution_8: "f32[64, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_6, primals_25, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        add_42: "i64[]" = torch.ops.aten.add.Tensor(primals_188, 1)
        var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
        getitem_18: "f32[1, 64, 1, 1]" = var_mean_8[0]
        getitem_19: "f32[1, 64, 1, 1]" = var_mean_8[1];  var_mean_8 = None
        add_43: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
        rsqrt_8: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_8: "f32[64, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_19)
        mul_56: "f32[64, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_25: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_57: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
        mul_58: "f32[64]" = torch.ops.aten.mul.Tensor(primals_186, 0.9)
        add_44: "f32[64]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
        squeeze_26: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_59: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000049824865598);  squeeze_26 = None
        mul_60: "f32[64]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
        mul_61: "f32[64]" = torch.ops.aten.mul.Tensor(primals_187, 0.9)
        add_45: "f32[64]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
        unsqueeze_32: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1)
        unsqueeze_33: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_62: "f32[64, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        unsqueeze_34: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1);  primals_27 = None
        unsqueeze_35: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_46: "f32[64, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:153, code: out = self.relu(out)
        relu_7: "f32[64, 64, 56, 56]" = torch.ops.aten.relu.default(add_46);  add_46 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:155, code: out = self.conv2(out)
        convolution_9: "f32[64, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_7, primals_28, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        add_47: "i64[]" = torch.ops.aten.add.Tensor(primals_191, 1)
        var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
        getitem_20: "f32[1, 64, 1, 1]" = var_mean_9[0]
        getitem_21: "f32[1, 64, 1, 1]" = var_mean_9[1];  var_mean_9 = None
        add_48: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
        rsqrt_9: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_9: "f32[64, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_21)
        mul_63: "f32[64, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_28: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_64: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
        mul_65: "f32[64]" = torch.ops.aten.mul.Tensor(primals_189, 0.9)
        add_49: "f32[64]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        squeeze_29: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_66: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000049824865598);  squeeze_29 = None
        mul_67: "f32[64]" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
        mul_68: "f32[64]" = torch.ops.aten.mul.Tensor(primals_190, 0.9)
        add_50: "f32[64]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        unsqueeze_36: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
        unsqueeze_37: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_69: "f32[64, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
        unsqueeze_38: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
        unsqueeze_39: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_51: "f32[64, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:157, code: out = self.relu(out)
        relu_8: "f32[64, 64, 56, 56]" = torch.ops.aten.relu.default(add_51);  add_51 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:159, code: out = self.conv3(out)
        convolution_10: "f32[64, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_8, primals_31, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        add_52: "i64[]" = torch.ops.aten.add.Tensor(primals_194, 1)
        var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
        getitem_22: "f32[1, 256, 1, 1]" = var_mean_10[0]
        getitem_23: "f32[1, 256, 1, 1]" = var_mean_10[1];  var_mean_10 = None
        add_53: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
        rsqrt_10: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_10: "f32[64, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_23)
        mul_70: "f32[64, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_31: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_71: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
        mul_72: "f32[256]" = torch.ops.aten.mul.Tensor(primals_192, 0.9)
        add_54: "f32[256]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
        squeeze_32: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_73: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000049824865598);  squeeze_32 = None
        mul_74: "f32[256]" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
        mul_75: "f32[256]" = torch.ops.aten.mul.Tensor(primals_193, 0.9)
        add_55: "f32[256]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        unsqueeze_40: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1)
        unsqueeze_41: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_76: "f32[64, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
        unsqueeze_42: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1);  primals_33 = None
        unsqueeze_43: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_56: "f32[64, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:166, code: out += residual
        add_57: "f32[64, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_56, relu_6);  add_56 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:173, code: out = self.relu(out)
        relu_9: "f32[64, 256, 56, 56]" = torch.ops.aten.relu.default(add_57);  add_57 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:151, code: out = self.conv1(x)
        convolution_11: "f32[64, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_9, primals_34, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        add_58: "i64[]" = torch.ops.aten.add.Tensor(primals_197, 1)
        var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
        getitem_24: "f32[1, 128, 1, 1]" = var_mean_11[0]
        getitem_25: "f32[1, 128, 1, 1]" = var_mean_11[1];  var_mean_11 = None
        add_59: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
        rsqrt_11: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        sub_11: "f32[64, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_25)
        mul_77: "f32[64, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_33: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_34: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_78: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
        mul_79: "f32[128]" = torch.ops.aten.mul.Tensor(primals_195, 0.9)
        add_60: "f32[128]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
        squeeze_35: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_80: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000049824865598);  squeeze_35 = None
        mul_81: "f32[128]" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
        mul_82: "f32[128]" = torch.ops.aten.mul.Tensor(primals_196, 0.9)
        add_61: "f32[128]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
        unsqueeze_44: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
        unsqueeze_45: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_83: "f32[64, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
        unsqueeze_46: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
        unsqueeze_47: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_62: "f32[64, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:153, code: out = self.relu(out)
        relu_10: "f32[64, 128, 56, 56]" = torch.ops.aten.relu.default(add_62);  add_62 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:155, code: out = self.conv2(out)
        convolution_12: "f32[64, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_10, primals_37, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        add_63: "i64[]" = torch.ops.aten.add.Tensor(primals_200, 1)
        var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
        getitem_26: "f32[1, 128, 1, 1]" = var_mean_12[0]
        getitem_27: "f32[1, 128, 1, 1]" = var_mean_12[1];  var_mean_12 = None
        add_64: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
        rsqrt_12: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_12: "f32[64, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_27)
        mul_84: "f32[64, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_36: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_37: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_85: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
        mul_86: "f32[128]" = torch.ops.aten.mul.Tensor(primals_198, 0.9)
        add_65: "f32[128]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
        squeeze_38: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_87: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000199302441455);  squeeze_38 = None
        mul_88: "f32[128]" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
        mul_89: "f32[128]" = torch.ops.aten.mul.Tensor(primals_199, 0.9)
        add_66: "f32[128]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
        unsqueeze_48: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1)
        unsqueeze_49: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_90: "f32[64, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
        unsqueeze_50: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1);  primals_39 = None
        unsqueeze_51: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_67: "f32[64, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:157, code: out = self.relu(out)
        relu_11: "f32[64, 128, 28, 28]" = torch.ops.aten.relu.default(add_67);  add_67 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:159, code: out = self.conv3(out)
        convolution_13: "f32[64, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_11, primals_40, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        add_68: "i64[]" = torch.ops.aten.add.Tensor(primals_203, 1)
        var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
        getitem_28: "f32[1, 512, 1, 1]" = var_mean_13[0]
        getitem_29: "f32[1, 512, 1, 1]" = var_mean_13[1];  var_mean_13 = None
        add_69: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
        rsqrt_13: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_13: "f32[64, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_29)
        mul_91: "f32[64, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_39: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_40: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_92: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
        mul_93: "f32[512]" = torch.ops.aten.mul.Tensor(primals_201, 0.9)
        add_70: "f32[512]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        squeeze_41: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_94: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0000199302441455);  squeeze_41 = None
        mul_95: "f32[512]" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
        mul_96: "f32[512]" = torch.ops.aten.mul.Tensor(primals_202, 0.9)
        add_71: "f32[512]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
        unsqueeze_52: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
        unsqueeze_53: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_97: "f32[64, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
        unsqueeze_54: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
        unsqueeze_55: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_72: "f32[64, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:163, code: residual = self.downsample(x)
        convolution_14: "f32[64, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_9, primals_43, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_73: "i64[]" = torch.ops.aten.add.Tensor(primals_206, 1)
        var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
        getitem_30: "f32[1, 512, 1, 1]" = var_mean_14[0]
        getitem_31: "f32[1, 512, 1, 1]" = var_mean_14[1];  var_mean_14 = None
        add_74: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
        rsqrt_14: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_14: "f32[64, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_31)
        mul_98: "f32[64, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_42: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_43: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_99: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
        mul_100: "f32[512]" = torch.ops.aten.mul.Tensor(primals_204, 0.9)
        add_75: "f32[512]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        squeeze_44: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_101: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0000199302441455);  squeeze_44 = None
        mul_102: "f32[512]" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
        mul_103: "f32[512]" = torch.ops.aten.mul.Tensor(primals_205, 0.9)
        add_76: "f32[512]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        unsqueeze_56: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1)
        unsqueeze_57: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_104: "f32[64, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        unsqueeze_58: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1);  primals_45 = None
        unsqueeze_59: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_77: "f32[64, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:166, code: out += residual
        add_78: "f32[64, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_72, add_77);  add_72 = add_77 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:173, code: out = self.relu(out)
        relu_12: "f32[64, 512, 28, 28]" = torch.ops.aten.relu.default(add_78);  add_78 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:151, code: out = self.conv1(x)
        convolution_15: "f32[64, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_12, primals_46, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        add_79: "i64[]" = torch.ops.aten.add.Tensor(primals_209, 1)
        var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
        getitem_32: "f32[1, 128, 1, 1]" = var_mean_15[0]
        getitem_33: "f32[1, 128, 1, 1]" = var_mean_15[1];  var_mean_15 = None
        add_80: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
        rsqrt_15: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_15: "f32[64, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_33)
        mul_105: "f32[64, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_45: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_46: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_106: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
        mul_107: "f32[128]" = torch.ops.aten.mul.Tensor(primals_207, 0.9)
        add_81: "f32[128]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
        squeeze_47: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_108: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0000199302441455);  squeeze_47 = None
        mul_109: "f32[128]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
        mul_110: "f32[128]" = torch.ops.aten.mul.Tensor(primals_208, 0.9)
        add_82: "f32[128]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
        unsqueeze_60: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
        unsqueeze_61: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_111: "f32[64, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
        unsqueeze_62: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
        unsqueeze_63: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_83: "f32[64, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:153, code: out = self.relu(out)
        relu_13: "f32[64, 128, 28, 28]" = torch.ops.aten.relu.default(add_83);  add_83 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:155, code: out = self.conv2(out)
        convolution_16: "f32[64, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_13, primals_49, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        add_84: "i64[]" = torch.ops.aten.add.Tensor(primals_212, 1)
        var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
        getitem_34: "f32[1, 128, 1, 1]" = var_mean_16[0]
        getitem_35: "f32[1, 128, 1, 1]" = var_mean_16[1];  var_mean_16 = None
        add_85: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
        rsqrt_16: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_16: "f32[64, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_35)
        mul_112: "f32[64, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_48: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_49: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_113: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
        mul_114: "f32[128]" = torch.ops.aten.mul.Tensor(primals_210, 0.9)
        add_86: "f32[128]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        squeeze_50: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_115: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0000199302441455);  squeeze_50 = None
        mul_116: "f32[128]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
        mul_117: "f32[128]" = torch.ops.aten.mul.Tensor(primals_211, 0.9)
        add_87: "f32[128]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        unsqueeze_64: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1)
        unsqueeze_65: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_118: "f32[64, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
        unsqueeze_66: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1);  primals_51 = None
        unsqueeze_67: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_88: "f32[64, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:157, code: out = self.relu(out)
        relu_14: "f32[64, 128, 28, 28]" = torch.ops.aten.relu.default(add_88);  add_88 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:159, code: out = self.conv3(out)
        convolution_17: "f32[64, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_14, primals_52, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        add_89: "i64[]" = torch.ops.aten.add.Tensor(primals_215, 1)
        var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
        getitem_36: "f32[1, 512, 1, 1]" = var_mean_17[0]
        getitem_37: "f32[1, 512, 1, 1]" = var_mean_17[1];  var_mean_17 = None
        add_90: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
        rsqrt_17: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_17: "f32[64, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_37)
        mul_119: "f32[64, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        squeeze_51: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_52: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_120: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
        mul_121: "f32[512]" = torch.ops.aten.mul.Tensor(primals_213, 0.9)
        add_91: "f32[512]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
        squeeze_53: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_122: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0000199302441455);  squeeze_53 = None
        mul_123: "f32[512]" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
        mul_124: "f32[512]" = torch.ops.aten.mul.Tensor(primals_214, 0.9)
        add_92: "f32[512]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
        unsqueeze_68: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
        unsqueeze_69: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_125: "f32[64, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
        unsqueeze_70: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
        unsqueeze_71: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_93: "f32[64, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:166, code: out += residual
        add_94: "f32[64, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_93, relu_12);  add_93 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:173, code: out = self.relu(out)
        relu_15: "f32[64, 512, 28, 28]" = torch.ops.aten.relu.default(add_94);  add_94 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:151, code: out = self.conv1(x)
        convolution_18: "f32[64, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_15, primals_55, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        add_95: "i64[]" = torch.ops.aten.add.Tensor(primals_218, 1)
        var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
        getitem_38: "f32[1, 128, 1, 1]" = var_mean_18[0]
        getitem_39: "f32[1, 128, 1, 1]" = var_mean_18[1];  var_mean_18 = None
        add_96: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
        rsqrt_18: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_18: "f32[64, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_39)
        mul_126: "f32[64, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        squeeze_54: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_55: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_127: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
        mul_128: "f32[128]" = torch.ops.aten.mul.Tensor(primals_216, 0.9)
        add_97: "f32[128]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
        squeeze_56: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_129: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0000199302441455);  squeeze_56 = None
        mul_130: "f32[128]" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
        mul_131: "f32[128]" = torch.ops.aten.mul.Tensor(primals_217, 0.9)
        add_98: "f32[128]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
        unsqueeze_72: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
        unsqueeze_73: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_132: "f32[64, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
        unsqueeze_74: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
        unsqueeze_75: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_99: "f32[64, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:153, code: out = self.relu(out)
        relu_16: "f32[64, 128, 28, 28]" = torch.ops.aten.relu.default(add_99);  add_99 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:155, code: out = self.conv2(out)
        convolution_19: "f32[64, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_16, primals_58, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        add_100: "i64[]" = torch.ops.aten.add.Tensor(primals_221, 1)
        var_mean_19 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
        getitem_40: "f32[1, 128, 1, 1]" = var_mean_19[0]
        getitem_41: "f32[1, 128, 1, 1]" = var_mean_19[1];  var_mean_19 = None
        add_101: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
        rsqrt_19: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        sub_19: "f32[64, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_41)
        mul_133: "f32[64, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        squeeze_57: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_58: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_134: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
        mul_135: "f32[128]" = torch.ops.aten.mul.Tensor(primals_219, 0.9)
        add_102: "f32[128]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
        squeeze_59: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_136: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0000199302441455);  squeeze_59 = None
        mul_137: "f32[128]" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
        mul_138: "f32[128]" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
        add_103: "f32[128]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
        unsqueeze_76: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
        unsqueeze_77: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_139: "f32[64, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
        unsqueeze_78: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
        unsqueeze_79: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_104: "f32[64, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:157, code: out = self.relu(out)
        relu_17: "f32[64, 128, 28, 28]" = torch.ops.aten.relu.default(add_104);  add_104 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:159, code: out = self.conv3(out)
        convolution_20: "f32[64, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_17, primals_61, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        add_105: "i64[]" = torch.ops.aten.add.Tensor(primals_224, 1)
        var_mean_20 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
        getitem_42: "f32[1, 512, 1, 1]" = var_mean_20[0]
        getitem_43: "f32[1, 512, 1, 1]" = var_mean_20[1];  var_mean_20 = None
        add_106: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
        rsqrt_20: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_20: "f32[64, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_43)
        mul_140: "f32[64, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        squeeze_60: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        squeeze_61: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_141: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
        mul_142: "f32[512]" = torch.ops.aten.mul.Tensor(primals_222, 0.9)
        add_107: "f32[512]" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
        squeeze_62: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_143: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0000199302441455);  squeeze_62 = None
        mul_144: "f32[512]" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
        mul_145: "f32[512]" = torch.ops.aten.mul.Tensor(primals_223, 0.9)
        add_108: "f32[512]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        unsqueeze_80: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1)
        unsqueeze_81: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        mul_146: "f32[64, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
        unsqueeze_82: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1);  primals_63 = None
        unsqueeze_83: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        add_109: "f32[64, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:166, code: out += residual
        add_110: "f32[64, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_109, relu_15);  add_109 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:173, code: out = self.relu(out)
        relu_18: "f32[64, 512, 28, 28]" = torch.ops.aten.relu.default(add_110);  add_110 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:151, code: out = self.conv1(x)
        convolution_21: "f32[64, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_18, primals_64, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        add_111: "i64[]" = torch.ops.aten.add.Tensor(primals_227, 1)
        var_mean_21 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
        getitem_44: "f32[1, 128, 1, 1]" = var_mean_21[0]
        getitem_45: "f32[1, 128, 1, 1]" = var_mean_21[1];  var_mean_21 = None
        add_112: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
        rsqrt_21: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        sub_21: "f32[64, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_45)
        mul_147: "f32[64, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        squeeze_63: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        squeeze_64: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_148: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
        mul_149: "f32[128]" = torch.ops.aten.mul.Tensor(primals_225, 0.9)
        add_113: "f32[128]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
        squeeze_65: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_150: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0000199302441455);  squeeze_65 = None
        mul_151: "f32[128]" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
        mul_152: "f32[128]" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
        add_114: "f32[128]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
        unsqueeze_84: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
        unsqueeze_85: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_153: "f32[64, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
        unsqueeze_86: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
        unsqueeze_87: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_115: "f32[64, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:153, code: out = self.relu(out)
        relu_19: "f32[64, 128, 28, 28]" = torch.ops.aten.relu.default(add_115);  add_115 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:155, code: out = self.conv2(out)
        convolution_22: "f32[64, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_19, primals_67, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        add_116: "i64[]" = torch.ops.aten.add.Tensor(primals_230, 1)
        var_mean_22 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
        getitem_46: "f32[1, 128, 1, 1]" = var_mean_22[0]
        getitem_47: "f32[1, 128, 1, 1]" = var_mean_22[1];  var_mean_22 = None
        add_117: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
        rsqrt_22: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
        sub_22: "f32[64, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_47)
        mul_154: "f32[64, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        squeeze_66: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_67: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_155: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
        mul_156: "f32[128]" = torch.ops.aten.mul.Tensor(primals_228, 0.9)
        add_118: "f32[128]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
        squeeze_68: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_157: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0000199302441455);  squeeze_68 = None
        mul_158: "f32[128]" = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
        mul_159: "f32[128]" = torch.ops.aten.mul.Tensor(primals_229, 0.9)
        add_119: "f32[128]" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
        unsqueeze_88: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1)
        unsqueeze_89: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_160: "f32[64, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
        unsqueeze_90: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1);  primals_69 = None
        unsqueeze_91: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_120: "f32[64, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:157, code: out = self.relu(out)
        relu_20: "f32[64, 128, 28, 28]" = torch.ops.aten.relu.default(add_120);  add_120 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:159, code: out = self.conv3(out)
        convolution_23: "f32[64, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_20, primals_70, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        add_121: "i64[]" = torch.ops.aten.add.Tensor(primals_233, 1)
        var_mean_23 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
        getitem_48: "f32[1, 512, 1, 1]" = var_mean_23[0]
        getitem_49: "f32[1, 512, 1, 1]" = var_mean_23[1];  var_mean_23 = None
        add_122: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
        rsqrt_23: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_23: "f32[64, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_49)
        mul_161: "f32[64, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        squeeze_69: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        squeeze_70: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_162: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
        mul_163: "f32[512]" = torch.ops.aten.mul.Tensor(primals_231, 0.9)
        add_123: "f32[512]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        squeeze_71: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_164: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0000199302441455);  squeeze_71 = None
        mul_165: "f32[512]" = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
        mul_166: "f32[512]" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
        add_124: "f32[512]" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
        unsqueeze_92: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
        unsqueeze_93: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_167: "f32[64, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_93);  mul_161 = unsqueeze_93 = None
        unsqueeze_94: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
        unsqueeze_95: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_125: "f32[64, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_95);  mul_167 = unsqueeze_95 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:166, code: out += residual
        add_126: "f32[64, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_125, relu_18);  add_125 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:173, code: out = self.relu(out)
        relu_21: "f32[64, 512, 28, 28]" = torch.ops.aten.relu.default(add_126);  add_126 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:151, code: out = self.conv1(x)
        convolution_24: "f32[64, 256, 28, 28]" = torch.ops.aten.convolution.default(relu_21, primals_73, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        add_127: "i64[]" = torch.ops.aten.add.Tensor(primals_236, 1)
        var_mean_24 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
        getitem_50: "f32[1, 256, 1, 1]" = var_mean_24[0]
        getitem_51: "f32[1, 256, 1, 1]" = var_mean_24[1];  var_mean_24 = None
        add_128: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
        rsqrt_24: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        sub_24: "f32[64, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_51)
        mul_168: "f32[64, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        squeeze_72: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        squeeze_73: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_169: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
        mul_170: "f32[256]" = torch.ops.aten.mul.Tensor(primals_234, 0.9)
        add_129: "f32[256]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
        squeeze_74: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_171: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0000199302441455);  squeeze_74 = None
        mul_172: "f32[256]" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
        mul_173: "f32[256]" = torch.ops.aten.mul.Tensor(primals_235, 0.9)
        add_130: "f32[256]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
        unsqueeze_96: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1)
        unsqueeze_97: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_174: "f32[64, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
        unsqueeze_98: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1);  primals_75 = None
        unsqueeze_99: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_131: "f32[64, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:153, code: out = self.relu(out)
        relu_22: "f32[64, 256, 28, 28]" = torch.ops.aten.relu.default(add_131);  add_131 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:155, code: out = self.conv2(out)
        convolution_25: "f32[64, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_22, primals_76, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        add_132: "i64[]" = torch.ops.aten.add.Tensor(primals_239, 1)
        var_mean_25 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
        getitem_52: "f32[1, 256, 1, 1]" = var_mean_25[0]
        getitem_53: "f32[1, 256, 1, 1]" = var_mean_25[1];  var_mean_25 = None
        add_133: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
        rsqrt_25: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        sub_25: "f32[64, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_53)
        mul_175: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        squeeze_75: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_76: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_176: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
        mul_177: "f32[256]" = torch.ops.aten.mul.Tensor(primals_237, 0.9)
        add_134: "f32[256]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
        squeeze_77: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_178: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0000797257434426);  squeeze_77 = None
        mul_179: "f32[256]" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
        mul_180: "f32[256]" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
        add_135: "f32[256]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
        unsqueeze_100: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
        unsqueeze_101: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_181: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
        unsqueeze_102: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
        unsqueeze_103: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_136: "f32[64, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:157, code: out = self.relu(out)
        relu_23: "f32[64, 256, 14, 14]" = torch.ops.aten.relu.default(add_136);  add_136 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:159, code: out = self.conv3(out)
        convolution_26: "f32[64, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_23, primals_79, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        add_137: "i64[]" = torch.ops.aten.add.Tensor(primals_242, 1)
        var_mean_26 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
        getitem_54: "f32[1, 1024, 1, 1]" = var_mean_26[0]
        getitem_55: "f32[1, 1024, 1, 1]" = var_mean_26[1];  var_mean_26 = None
        add_138: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
        rsqrt_26: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        sub_26: "f32[64, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_55)
        mul_182: "f32[64, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        squeeze_78: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        squeeze_79: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_183: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
        mul_184: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_240, 0.9)
        add_139: "f32[1024]" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
        squeeze_80: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_185: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0000797257434426);  squeeze_80 = None
        mul_186: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
        mul_187: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_241, 0.9)
        add_140: "f32[1024]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
        unsqueeze_104: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1)
        unsqueeze_105: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        mul_188: "f32[64, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
        unsqueeze_106: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1);  primals_81 = None
        unsqueeze_107: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        add_141: "f32[64, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:163, code: residual = self.downsample(x)
        convolution_27: "f32[64, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_21, primals_82, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_142: "i64[]" = torch.ops.aten.add.Tensor(primals_245, 1)
        var_mean_27 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
        getitem_56: "f32[1, 1024, 1, 1]" = var_mean_27[0]
        getitem_57: "f32[1, 1024, 1, 1]" = var_mean_27[1];  var_mean_27 = None
        add_143: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
        rsqrt_27: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
        sub_27: "f32[64, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_57)
        mul_189: "f32[64, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        squeeze_81: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        squeeze_82: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_190: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
        mul_191: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_243, 0.9)
        add_144: "f32[1024]" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
        squeeze_83: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        mul_192: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0000797257434426);  squeeze_83 = None
        mul_193: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
        mul_194: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
        add_145: "f32[1024]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
        unsqueeze_108: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
        unsqueeze_109: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_195: "f32[64, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
        unsqueeze_110: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
        unsqueeze_111: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_146: "f32[64, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:166, code: out += residual
        add_147: "f32[64, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_141, add_146);  add_141 = add_146 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:173, code: out = self.relu(out)
        relu_24: "f32[64, 1024, 14, 14]" = torch.ops.aten.relu.default(add_147);  add_147 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:151, code: out = self.conv1(x)
        convolution_28: "f32[64, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_24, primals_85, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        add_148: "i64[]" = torch.ops.aten.add.Tensor(primals_248, 1)
        var_mean_28 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
        getitem_58: "f32[1, 256, 1, 1]" = var_mean_28[0]
        getitem_59: "f32[1, 256, 1, 1]" = var_mean_28[1];  var_mean_28 = None
        add_149: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
        rsqrt_28: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        sub_28: "f32[64, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_59)
        mul_196: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        squeeze_84: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        squeeze_85: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_197: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
        mul_198: "f32[256]" = torch.ops.aten.mul.Tensor(primals_246, 0.9)
        add_150: "f32[256]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
        squeeze_86: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_199: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0000797257434426);  squeeze_86 = None
        mul_200: "f32[256]" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
        mul_201: "f32[256]" = torch.ops.aten.mul.Tensor(primals_247, 0.9)
        add_151: "f32[256]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
        unsqueeze_112: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1)
        unsqueeze_113: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_202: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
        unsqueeze_114: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1);  primals_87 = None
        unsqueeze_115: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_152: "f32[64, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:153, code: out = self.relu(out)
        relu_25: "f32[64, 256, 14, 14]" = torch.ops.aten.relu.default(add_152);  add_152 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:155, code: out = self.conv2(out)
        convolution_29: "f32[64, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_25, primals_88, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        add_153: "i64[]" = torch.ops.aten.add.Tensor(primals_251, 1)
        var_mean_29 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
        getitem_60: "f32[1, 256, 1, 1]" = var_mean_29[0]
        getitem_61: "f32[1, 256, 1, 1]" = var_mean_29[1];  var_mean_29 = None
        add_154: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
        rsqrt_29: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        sub_29: "f32[64, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_61)
        mul_203: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        squeeze_87: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        squeeze_88: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_204: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
        mul_205: "f32[256]" = torch.ops.aten.mul.Tensor(primals_249, 0.9)
        add_155: "f32[256]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
        squeeze_89: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_206: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0000797257434426);  squeeze_89 = None
        mul_207: "f32[256]" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
        mul_208: "f32[256]" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
        add_156: "f32[256]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
        unsqueeze_116: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
        unsqueeze_117: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_209: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
        unsqueeze_118: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
        unsqueeze_119: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_157: "f32[64, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:157, code: out = self.relu(out)
        relu_26: "f32[64, 256, 14, 14]" = torch.ops.aten.relu.default(add_157);  add_157 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:159, code: out = self.conv3(out)
        convolution_30: "f32[64, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_26, primals_91, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        add_158: "i64[]" = torch.ops.aten.add.Tensor(primals_254, 1)
        var_mean_30 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
        getitem_62: "f32[1, 1024, 1, 1]" = var_mean_30[0]
        getitem_63: "f32[1, 1024, 1, 1]" = var_mean_30[1];  var_mean_30 = None
        add_159: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
        rsqrt_30: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        sub_30: "f32[64, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_63)
        mul_210: "f32[64, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        squeeze_90: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        squeeze_91: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_211: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
        mul_212: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_252, 0.9)
        add_160: "f32[1024]" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
        squeeze_92: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_213: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0000797257434426);  squeeze_92 = None
        mul_214: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
        mul_215: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_253, 0.9)
        add_161: "f32[1024]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
        unsqueeze_120: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1)
        unsqueeze_121: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        mul_216: "f32[64, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
        unsqueeze_122: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1);  primals_93 = None
        unsqueeze_123: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        add_162: "f32[64, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:166, code: out += residual
        add_163: "f32[64, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_162, relu_24);  add_162 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:173, code: out = self.relu(out)
        relu_27: "f32[64, 1024, 14, 14]" = torch.ops.aten.relu.default(add_163);  add_163 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:151, code: out = self.conv1(x)
        convolution_31: "f32[64, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_27, primals_94, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        add_164: "i64[]" = torch.ops.aten.add.Tensor(primals_257, 1)
        var_mean_31 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
        getitem_64: "f32[1, 256, 1, 1]" = var_mean_31[0]
        getitem_65: "f32[1, 256, 1, 1]" = var_mean_31[1];  var_mean_31 = None
        add_165: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
        rsqrt_31: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        sub_31: "f32[64, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_65)
        mul_217: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        squeeze_93: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        squeeze_94: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_218: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
        mul_219: "f32[256]" = torch.ops.aten.mul.Tensor(primals_255, 0.9)
        add_166: "f32[256]" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
        squeeze_95: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_220: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0000797257434426);  squeeze_95 = None
        mul_221: "f32[256]" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
        mul_222: "f32[256]" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
        add_167: "f32[256]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
        unsqueeze_124: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
        unsqueeze_125: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_223: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
        unsqueeze_126: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
        unsqueeze_127: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_168: "f32[64, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:153, code: out = self.relu(out)
        relu_28: "f32[64, 256, 14, 14]" = torch.ops.aten.relu.default(add_168);  add_168 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:155, code: out = self.conv2(out)
        convolution_32: "f32[64, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_28, primals_97, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        add_169: "i64[]" = torch.ops.aten.add.Tensor(primals_260, 1)
        var_mean_32 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
        getitem_66: "f32[1, 256, 1, 1]" = var_mean_32[0]
        getitem_67: "f32[1, 256, 1, 1]" = var_mean_32[1];  var_mean_32 = None
        add_170: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
        rsqrt_32: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_32: "f32[64, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_67)
        mul_224: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        squeeze_96: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        squeeze_97: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_225: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
        mul_226: "f32[256]" = torch.ops.aten.mul.Tensor(primals_258, 0.9)
        add_171: "f32[256]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
        squeeze_98: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_227: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0000797257434426);  squeeze_98 = None
        mul_228: "f32[256]" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
        mul_229: "f32[256]" = torch.ops.aten.mul.Tensor(primals_259, 0.9)
        add_172: "f32[256]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
        unsqueeze_128: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
        unsqueeze_129: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        mul_230: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_129);  mul_224 = unsqueeze_129 = None
        unsqueeze_130: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
        unsqueeze_131: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        add_173: "f32[64, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_131);  mul_230 = unsqueeze_131 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:157, code: out = self.relu(out)
        relu_29: "f32[64, 256, 14, 14]" = torch.ops.aten.relu.default(add_173);  add_173 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:159, code: out = self.conv3(out)
        convolution_33: "f32[64, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_29, primals_100, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        add_174: "i64[]" = torch.ops.aten.add.Tensor(primals_263, 1)
        var_mean_33 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
        getitem_68: "f32[1, 1024, 1, 1]" = var_mean_33[0]
        getitem_69: "f32[1, 1024, 1, 1]" = var_mean_33[1];  var_mean_33 = None
        add_175: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
        rsqrt_33: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
        sub_33: "f32[64, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_69)
        mul_231: "f32[64, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        squeeze_99: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        squeeze_100: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_232: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
        mul_233: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_261, 0.9)
        add_176: "f32[1024]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
        squeeze_101: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_234: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0000797257434426);  squeeze_101 = None
        mul_235: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
        mul_236: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
        add_177: "f32[1024]" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
        unsqueeze_132: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
        unsqueeze_133: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_237: "f32[64, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
        unsqueeze_134: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
        unsqueeze_135: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_178: "f32[64, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:166, code: out += residual
        add_179: "f32[64, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_178, relu_27);  add_178 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:173, code: out = self.relu(out)
        relu_30: "f32[64, 1024, 14, 14]" = torch.ops.aten.relu.default(add_179);  add_179 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:151, code: out = self.conv1(x)
        convolution_34: "f32[64, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_30, primals_103, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        add_180: "i64[]" = torch.ops.aten.add.Tensor(primals_266, 1)
        var_mean_34 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
        getitem_70: "f32[1, 256, 1, 1]" = var_mean_34[0]
        getitem_71: "f32[1, 256, 1, 1]" = var_mean_34[1];  var_mean_34 = None
        add_181: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
        rsqrt_34: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_34: "f32[64, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_71)
        mul_238: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        squeeze_102: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        squeeze_103: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_239: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
        mul_240: "f32[256]" = torch.ops.aten.mul.Tensor(primals_264, 0.9)
        add_182: "f32[256]" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
        squeeze_104: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_241: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0000797257434426);  squeeze_104 = None
        mul_242: "f32[256]" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
        mul_243: "f32[256]" = torch.ops.aten.mul.Tensor(primals_265, 0.9)
        add_183: "f32[256]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
        unsqueeze_136: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1)
        unsqueeze_137: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        mul_244: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
        unsqueeze_138: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1);  primals_105 = None
        unsqueeze_139: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        add_184: "f32[64, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:153, code: out = self.relu(out)
        relu_31: "f32[64, 256, 14, 14]" = torch.ops.aten.relu.default(add_184);  add_184 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:155, code: out = self.conv2(out)
        convolution_35: "f32[64, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_31, primals_106, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        add_185: "i64[]" = torch.ops.aten.add.Tensor(primals_269, 1)
        var_mean_35 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
        getitem_72: "f32[1, 256, 1, 1]" = var_mean_35[0]
        getitem_73: "f32[1, 256, 1, 1]" = var_mean_35[1];  var_mean_35 = None
        add_186: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
        rsqrt_35: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_35: "f32[64, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_73)
        mul_245: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        squeeze_105: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        squeeze_106: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
        mul_246: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
        mul_247: "f32[256]" = torch.ops.aten.mul.Tensor(primals_267, 0.9)
        add_187: "f32[256]" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
        squeeze_107: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        mul_248: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0000797257434426);  squeeze_107 = None
        mul_249: "f32[256]" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
        mul_250: "f32[256]" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
        add_188: "f32[256]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
        unsqueeze_140: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
        unsqueeze_141: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_251: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_141);  mul_245 = unsqueeze_141 = None
        unsqueeze_142: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
        unsqueeze_143: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_189: "f32[64, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_143);  mul_251 = unsqueeze_143 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:157, code: out = self.relu(out)
        relu_32: "f32[64, 256, 14, 14]" = torch.ops.aten.relu.default(add_189);  add_189 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:159, code: out = self.conv3(out)
        convolution_36: "f32[64, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_32, primals_109, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        add_190: "i64[]" = torch.ops.aten.add.Tensor(primals_272, 1)
        var_mean_36 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
        getitem_74: "f32[1, 1024, 1, 1]" = var_mean_36[0]
        getitem_75: "f32[1, 1024, 1, 1]" = var_mean_36[1];  var_mean_36 = None
        add_191: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
        rsqrt_36: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        sub_36: "f32[64, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_75)
        mul_252: "f32[64, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        squeeze_108: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        squeeze_109: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_253: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
        mul_254: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_270, 0.9)
        add_192: "f32[1024]" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
        squeeze_110: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_255: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0000797257434426);  squeeze_110 = None
        mul_256: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
        mul_257: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_271, 0.9)
        add_193: "f32[1024]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
        unsqueeze_144: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1)
        unsqueeze_145: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        mul_258: "f32[64, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
        unsqueeze_146: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1);  primals_111 = None
        unsqueeze_147: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        add_194: "f32[64, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:166, code: out += residual
        add_195: "f32[64, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_194, relu_30);  add_194 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:173, code: out = self.relu(out)
        relu_33: "f32[64, 1024, 14, 14]" = torch.ops.aten.relu.default(add_195);  add_195 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:151, code: out = self.conv1(x)
        convolution_37: "f32[64, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_33, primals_112, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        add_196: "i64[]" = torch.ops.aten.add.Tensor(primals_275, 1)
        var_mean_37 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
        getitem_76: "f32[1, 256, 1, 1]" = var_mean_37[0]
        getitem_77: "f32[1, 256, 1, 1]" = var_mean_37[1];  var_mean_37 = None
        add_197: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
        rsqrt_37: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
        sub_37: "f32[64, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_77)
        mul_259: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        squeeze_111: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_112: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_260: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
        mul_261: "f32[256]" = torch.ops.aten.mul.Tensor(primals_273, 0.9)
        add_198: "f32[256]" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
        squeeze_113: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_262: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0000797257434426);  squeeze_113 = None
        mul_263: "f32[256]" = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
        mul_264: "f32[256]" = torch.ops.aten.mul.Tensor(primals_274, 0.9)
        add_199: "f32[256]" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
        unsqueeze_148: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
        unsqueeze_149: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_265: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
        unsqueeze_150: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
        unsqueeze_151: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_200: "f32[64, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:153, code: out = self.relu(out)
        relu_34: "f32[64, 256, 14, 14]" = torch.ops.aten.relu.default(add_200);  add_200 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:155, code: out = self.conv2(out)
        convolution_38: "f32[64, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_34, primals_115, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        add_201: "i64[]" = torch.ops.aten.add.Tensor(primals_278, 1)
        var_mean_38 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
        getitem_78: "f32[1, 256, 1, 1]" = var_mean_38[0]
        getitem_79: "f32[1, 256, 1, 1]" = var_mean_38[1];  var_mean_38 = None
        add_202: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
        rsqrt_38: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
        sub_38: "f32[64, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_79)
        mul_266: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        squeeze_114: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        squeeze_115: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_267: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
        mul_268: "f32[256]" = torch.ops.aten.mul.Tensor(primals_276, 0.9)
        add_203: "f32[256]" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
        squeeze_116: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_269: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0000797257434426);  squeeze_116 = None
        mul_270: "f32[256]" = torch.ops.aten.mul.Tensor(mul_269, 0.1);  mul_269 = None
        mul_271: "f32[256]" = torch.ops.aten.mul.Tensor(primals_277, 0.9)
        add_204: "f32[256]" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
        unsqueeze_152: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1)
        unsqueeze_153: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        mul_272: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
        unsqueeze_154: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1);  primals_117 = None
        unsqueeze_155: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        add_205: "f32[64, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:157, code: out = self.relu(out)
        relu_35: "f32[64, 256, 14, 14]" = torch.ops.aten.relu.default(add_205);  add_205 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:159, code: out = self.conv3(out)
        convolution_39: "f32[64, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_35, primals_118, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        add_206: "i64[]" = torch.ops.aten.add.Tensor(primals_281, 1)
        var_mean_39 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
        getitem_80: "f32[1, 1024, 1, 1]" = var_mean_39[0]
        getitem_81: "f32[1, 1024, 1, 1]" = var_mean_39[1];  var_mean_39 = None
        add_207: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
        rsqrt_39: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
        sub_39: "f32[64, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_81)
        mul_273: "f32[64, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        squeeze_117: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        squeeze_118: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_274: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
        mul_275: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_279, 0.9)
        add_208: "f32[1024]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
        squeeze_119: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
        mul_276: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0000797257434426);  squeeze_119 = None
        mul_277: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_276, 0.1);  mul_276 = None
        mul_278: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_280, 0.9)
        add_209: "f32[1024]" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
        unsqueeze_156: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
        unsqueeze_157: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_279: "f32[64, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
        unsqueeze_158: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
        unsqueeze_159: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_210: "f32[64, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:166, code: out += residual
        add_211: "f32[64, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_210, relu_33);  add_210 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:173, code: out = self.relu(out)
        relu_36: "f32[64, 1024, 14, 14]" = torch.ops.aten.relu.default(add_211);  add_211 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:151, code: out = self.conv1(x)
        convolution_40: "f32[64, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_36, primals_121, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        add_212: "i64[]" = torch.ops.aten.add.Tensor(primals_284, 1)
        var_mean_40 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
        getitem_82: "f32[1, 256, 1, 1]" = var_mean_40[0]
        getitem_83: "f32[1, 256, 1, 1]" = var_mean_40[1];  var_mean_40 = None
        add_213: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
        rsqrt_40: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
        sub_40: "f32[64, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_83)
        mul_280: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        squeeze_120: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        squeeze_121: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_281: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
        mul_282: "f32[256]" = torch.ops.aten.mul.Tensor(primals_282, 0.9)
        add_214: "f32[256]" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
        squeeze_122: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        mul_283: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0000797257434426);  squeeze_122 = None
        mul_284: "f32[256]" = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
        mul_285: "f32[256]" = torch.ops.aten.mul.Tensor(primals_283, 0.9)
        add_215: "f32[256]" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
        unsqueeze_160: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1)
        unsqueeze_161: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        mul_286: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
        unsqueeze_162: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1);  primals_123 = None
        unsqueeze_163: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        add_216: "f32[64, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:153, code: out = self.relu(out)
        relu_37: "f32[64, 256, 14, 14]" = torch.ops.aten.relu.default(add_216);  add_216 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:155, code: out = self.conv2(out)
        convolution_41: "f32[64, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_37, primals_124, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        add_217: "i64[]" = torch.ops.aten.add.Tensor(primals_287, 1)
        var_mean_41 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
        getitem_84: "f32[1, 256, 1, 1]" = var_mean_41[0]
        getitem_85: "f32[1, 256, 1, 1]" = var_mean_41[1];  var_mean_41 = None
        add_218: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
        rsqrt_41: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        sub_41: "f32[64, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_85)
        mul_287: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
        squeeze_123: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        squeeze_124: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_288: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
        mul_289: "f32[256]" = torch.ops.aten.mul.Tensor(primals_285, 0.9)
        add_219: "f32[256]" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
        squeeze_125: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
        mul_290: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0000797257434426);  squeeze_125 = None
        mul_291: "f32[256]" = torch.ops.aten.mul.Tensor(mul_290, 0.1);  mul_290 = None
        mul_292: "f32[256]" = torch.ops.aten.mul.Tensor(primals_286, 0.9)
        add_220: "f32[256]" = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
        unsqueeze_164: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1)
        unsqueeze_165: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        mul_293: "f32[64, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_165);  mul_287 = unsqueeze_165 = None
        unsqueeze_166: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
        unsqueeze_167: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        add_221: "f32[64, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_167);  mul_293 = unsqueeze_167 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:157, code: out = self.relu(out)
        relu_38: "f32[64, 256, 14, 14]" = torch.ops.aten.relu.default(add_221);  add_221 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:159, code: out = self.conv3(out)
        convolution_42: "f32[64, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_38, primals_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        add_222: "i64[]" = torch.ops.aten.add.Tensor(primals_290, 1)
        var_mean_42 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
        getitem_86: "f32[1, 1024, 1, 1]" = var_mean_42[0]
        getitem_87: "f32[1, 1024, 1, 1]" = var_mean_42[1];  var_mean_42 = None
        add_223: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
        rsqrt_42: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
        sub_42: "f32[64, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_87)
        mul_294: "f32[64, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        squeeze_126: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        squeeze_127: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_295: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
        mul_296: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_288, 0.9)
        add_224: "f32[1024]" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
        squeeze_128: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
        mul_297: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0000797257434426);  squeeze_128 = None
        mul_298: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
        mul_299: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_289, 0.9)
        add_225: "f32[1024]" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
        unsqueeze_168: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1)
        unsqueeze_169: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        mul_300: "f32[64, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
        unsqueeze_170: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1);  primals_129 = None
        unsqueeze_171: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        add_226: "f32[64, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:166, code: out += residual
        add_227: "f32[64, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_226, relu_36);  add_226 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:173, code: out = self.relu(out)
        relu_39: "f32[64, 1024, 14, 14]" = torch.ops.aten.relu.default(add_227);  add_227 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:151, code: out = self.conv1(x)
        convolution_43: "f32[64, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_39, primals_130, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        add_228: "i64[]" = torch.ops.aten.add.Tensor(primals_293, 1)
        var_mean_43 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
        getitem_88: "f32[1, 512, 1, 1]" = var_mean_43[0]
        getitem_89: "f32[1, 512, 1, 1]" = var_mean_43[1];  var_mean_43 = None
        add_229: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
        rsqrt_43: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_43: "f32[64, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_89)
        mul_301: "f32[64, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        squeeze_129: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
        squeeze_130: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
        mul_302: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
        mul_303: "f32[512]" = torch.ops.aten.mul.Tensor(primals_291, 0.9)
        add_230: "f32[512]" = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
        squeeze_131: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
        mul_304: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0000797257434426);  squeeze_131 = None
        mul_305: "f32[512]" = torch.ops.aten.mul.Tensor(mul_304, 0.1);  mul_304 = None
        mul_306: "f32[512]" = torch.ops.aten.mul.Tensor(primals_292, 0.9)
        add_231: "f32[512]" = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
        unsqueeze_172: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1)
        unsqueeze_173: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_307: "f32[64, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_173);  mul_301 = unsqueeze_173 = None
        unsqueeze_174: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_132, -1);  primals_132 = None
        unsqueeze_175: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_232: "f32[64, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_175);  mul_307 = unsqueeze_175 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:153, code: out = self.relu(out)
        relu_40: "f32[64, 512, 14, 14]" = torch.ops.aten.relu.default(add_232);  add_232 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:155, code: out = self.conv2(out)
        convolution_44: "f32[64, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_40, primals_133, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        add_233: "i64[]" = torch.ops.aten.add.Tensor(primals_296, 1)
        var_mean_44 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
        getitem_90: "f32[1, 512, 1, 1]" = var_mean_44[0]
        getitem_91: "f32[1, 512, 1, 1]" = var_mean_44[1];  var_mean_44 = None
        add_234: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
        rsqrt_44: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_234);  add_234 = None
        sub_44: "f32[64, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_91)
        mul_308: "f32[64, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
        squeeze_132: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        squeeze_133: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
        mul_309: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
        mul_310: "f32[512]" = torch.ops.aten.mul.Tensor(primals_294, 0.9)
        add_235: "f32[512]" = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
        squeeze_134: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
        mul_311: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0003189792663476);  squeeze_134 = None
        mul_312: "f32[512]" = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
        mul_313: "f32[512]" = torch.ops.aten.mul.Tensor(primals_295, 0.9)
        add_236: "f32[512]" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
        unsqueeze_176: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1)
        unsqueeze_177: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        mul_314: "f32[64, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_177);  mul_308 = unsqueeze_177 = None
        unsqueeze_178: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_135, -1);  primals_135 = None
        unsqueeze_179: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        add_237: "f32[64, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_179);  mul_314 = unsqueeze_179 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:157, code: out = self.relu(out)
        relu_41: "f32[64, 512, 7, 7]" = torch.ops.aten.relu.default(add_237);  add_237 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:159, code: out = self.conv3(out)
        convolution_45: "f32[64, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_41, primals_136, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        add_238: "i64[]" = torch.ops.aten.add.Tensor(primals_299, 1)
        var_mean_45 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
        getitem_92: "f32[1, 2048, 1, 1]" = var_mean_45[0]
        getitem_93: "f32[1, 2048, 1, 1]" = var_mean_45[1];  var_mean_45 = None
        add_239: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05)
        rsqrt_45: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        sub_45: "f32[64, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_93)
        mul_315: "f32[64, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        squeeze_135: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        squeeze_136: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_316: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
        mul_317: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_297, 0.9)
        add_240: "f32[2048]" = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
        squeeze_137: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
        mul_318: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0003189792663476);  squeeze_137 = None
        mul_319: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_318, 0.1);  mul_318 = None
        mul_320: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_298, 0.9)
        add_241: "f32[2048]" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
        unsqueeze_180: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_137, -1)
        unsqueeze_181: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_321: "f32[64, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
        unsqueeze_182: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_138, -1);  primals_138 = None
        unsqueeze_183: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_242: "f32[64, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:163, code: residual = self.downsample(x)
        convolution_46: "f32[64, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_39, primals_139, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_243: "i64[]" = torch.ops.aten.add.Tensor(primals_302, 1)
        var_mean_46 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
        getitem_94: "f32[1, 2048, 1, 1]" = var_mean_46[0]
        getitem_95: "f32[1, 2048, 1, 1]" = var_mean_46[1];  var_mean_46 = None
        add_244: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05)
        rsqrt_46: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
        sub_46: "f32[64, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_95)
        mul_322: "f32[64, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
        squeeze_138: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        squeeze_139: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
        mul_323: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
        mul_324: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_300, 0.9)
        add_245: "f32[2048]" = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
        squeeze_140: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
        mul_325: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0003189792663476);  squeeze_140 = None
        mul_326: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_325, 0.1);  mul_325 = None
        mul_327: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_301, 0.9)
        add_246: "f32[2048]" = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
        unsqueeze_184: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1)
        unsqueeze_185: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        mul_328: "f32[64, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_185);  mul_322 = unsqueeze_185 = None
        unsqueeze_186: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1);  primals_141 = None
        unsqueeze_187: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        add_247: "f32[64, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_328, unsqueeze_187);  mul_328 = unsqueeze_187 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:166, code: out += residual
        add_248: "f32[64, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_242, add_247);  add_242 = add_247 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:173, code: out = self.relu(out)
        relu_42: "f32[64, 2048, 7, 7]" = torch.ops.aten.relu.default(add_248);  add_248 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:151, code: out = self.conv1(x)
        convolution_47: "f32[64, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_42, primals_142, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        add_249: "i64[]" = torch.ops.aten.add.Tensor(primals_305, 1)
        var_mean_47 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
        getitem_96: "f32[1, 512, 1, 1]" = var_mean_47[0]
        getitem_97: "f32[1, 512, 1, 1]" = var_mean_47[1];  var_mean_47 = None
        add_250: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
        rsqrt_47: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
        sub_47: "f32[64, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_97)
        mul_329: "f32[64, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
        squeeze_141: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        squeeze_142: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
        mul_330: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
        mul_331: "f32[512]" = torch.ops.aten.mul.Tensor(primals_303, 0.9)
        add_251: "f32[512]" = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
        squeeze_143: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
        mul_332: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0003189792663476);  squeeze_143 = None
        mul_333: "f32[512]" = torch.ops.aten.mul.Tensor(mul_332, 0.1);  mul_332 = None
        mul_334: "f32[512]" = torch.ops.aten.mul.Tensor(primals_304, 0.9)
        add_252: "f32[512]" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
        unsqueeze_188: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_143, -1)
        unsqueeze_189: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_335: "f32[64, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_189);  mul_329 = unsqueeze_189 = None
        unsqueeze_190: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1);  primals_144 = None
        unsqueeze_191: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_253: "f32[64, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_191);  mul_335 = unsqueeze_191 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:153, code: out = self.relu(out)
        relu_43: "f32[64, 512, 7, 7]" = torch.ops.aten.relu.default(add_253);  add_253 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:155, code: out = self.conv2(out)
        convolution_48: "f32[64, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_43, primals_145, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        add_254: "i64[]" = torch.ops.aten.add.Tensor(primals_308, 1)
        var_mean_48 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
        getitem_98: "f32[1, 512, 1, 1]" = var_mean_48[0]
        getitem_99: "f32[1, 512, 1, 1]" = var_mean_48[1];  var_mean_48 = None
        add_255: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
        rsqrt_48: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_255);  add_255 = None
        sub_48: "f32[64, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_99)
        mul_336: "f32[64, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        squeeze_144: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
        squeeze_145: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_337: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
        mul_338: "f32[512]" = torch.ops.aten.mul.Tensor(primals_306, 0.9)
        add_256: "f32[512]" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
        squeeze_146: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
        mul_339: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0003189792663476);  squeeze_146 = None
        mul_340: "f32[512]" = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
        mul_341: "f32[512]" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
        add_257: "f32[512]" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
        unsqueeze_192: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_146, -1)
        unsqueeze_193: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        mul_342: "f32[64, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
        unsqueeze_194: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_147, -1);  primals_147 = None
        unsqueeze_195: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        add_258: "f32[64, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:157, code: out = self.relu(out)
        relu_44: "f32[64, 512, 7, 7]" = torch.ops.aten.relu.default(add_258);  add_258 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:159, code: out = self.conv3(out)
        convolution_49: "f32[64, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_44, primals_148, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        add_259: "i64[]" = torch.ops.aten.add.Tensor(primals_311, 1)
        var_mean_49 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
        getitem_100: "f32[1, 2048, 1, 1]" = var_mean_49[0]
        getitem_101: "f32[1, 2048, 1, 1]" = var_mean_49[1];  var_mean_49 = None
        add_260: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
        rsqrt_49: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_260);  add_260 = None
        sub_49: "f32[64, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_101)
        mul_343: "f32[64, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
        squeeze_147: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
        squeeze_148: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
        mul_344: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
        mul_345: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_309, 0.9)
        add_261: "f32[2048]" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
        squeeze_149: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
        mul_346: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0003189792663476);  squeeze_149 = None
        mul_347: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
        mul_348: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
        add_262: "f32[2048]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
        unsqueeze_196: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_149, -1)
        unsqueeze_197: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        mul_349: "f32[64, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_197);  mul_343 = unsqueeze_197 = None
        unsqueeze_198: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_150, -1);  primals_150 = None
        unsqueeze_199: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        add_263: "f32[64, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_199);  mul_349 = unsqueeze_199 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:166, code: out += residual
        add_264: "f32[64, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_263, relu_42);  add_263 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:173, code: out = self.relu(out)
        relu_45: "f32[64, 2048, 7, 7]" = torch.ops.aten.relu.default(add_264);  add_264 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:151, code: out = self.conv1(x)
        convolution_50: "f32[64, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_45, primals_151, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        add_265: "i64[]" = torch.ops.aten.add.Tensor(primals_314, 1)
        var_mean_50 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
        getitem_102: "f32[1, 512, 1, 1]" = var_mean_50[0]
        getitem_103: "f32[1, 512, 1, 1]" = var_mean_50[1];  var_mean_50 = None
        add_266: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
        rsqrt_50: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_266);  add_266 = None
        sub_50: "f32[64, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_103)
        mul_350: "f32[64, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
        squeeze_150: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        squeeze_151: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
        mul_351: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
        mul_352: "f32[512]" = torch.ops.aten.mul.Tensor(primals_312, 0.9)
        add_267: "f32[512]" = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
        squeeze_152: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
        mul_353: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0003189792663476);  squeeze_152 = None
        mul_354: "f32[512]" = torch.ops.aten.mul.Tensor(mul_353, 0.1);  mul_353 = None
        mul_355: "f32[512]" = torch.ops.aten.mul.Tensor(primals_313, 0.9)
        add_268: "f32[512]" = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
        unsqueeze_200: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_152, -1)
        unsqueeze_201: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        mul_356: "f32[64, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_201);  mul_350 = unsqueeze_201 = None
        unsqueeze_202: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_153, -1);  primals_153 = None
        unsqueeze_203: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        add_269: "f32[64, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_203);  mul_356 = unsqueeze_203 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:153, code: out = self.relu(out)
        relu_46: "f32[64, 512, 7, 7]" = torch.ops.aten.relu.default(add_269);  add_269 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:155, code: out = self.conv2(out)
        convolution_51: "f32[64, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_46, primals_154, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        add_270: "i64[]" = torch.ops.aten.add.Tensor(primals_317, 1)
        var_mean_51 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
        getitem_104: "f32[1, 512, 1, 1]" = var_mean_51[0]
        getitem_105: "f32[1, 512, 1, 1]" = var_mean_51[1];  var_mean_51 = None
        add_271: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
        rsqrt_51: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
        sub_51: "f32[64, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_105)
        mul_357: "f32[64, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
        squeeze_153: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
        squeeze_154: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_358: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
        mul_359: "f32[512]" = torch.ops.aten.mul.Tensor(primals_315, 0.9)
        add_272: "f32[512]" = torch.ops.aten.add.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
        squeeze_155: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
        mul_360: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0003189792663476);  squeeze_155 = None
        mul_361: "f32[512]" = torch.ops.aten.mul.Tensor(mul_360, 0.1);  mul_360 = None
        mul_362: "f32[512]" = torch.ops.aten.mul.Tensor(primals_316, 0.9)
        add_273: "f32[512]" = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
        unsqueeze_204: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_155, -1)
        unsqueeze_205: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_363: "f32[64, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
        unsqueeze_206: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1);  primals_156 = None
        unsqueeze_207: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_274: "f32[64, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:157, code: out = self.relu(out)
        relu_47: "f32[64, 512, 7, 7]" = torch.ops.aten.relu.default(add_274);  add_274 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:159, code: out = self.conv3(out)
        convolution_52: "f32[64, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_47, primals_157, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        add_275: "i64[]" = torch.ops.aten.add.Tensor(primals_320, 1)
        var_mean_52 = torch.ops.aten.var_mean.correction(convolution_52, [0, 2, 3], correction = 0, keepdim = True)
        getitem_106: "f32[1, 2048, 1, 1]" = var_mean_52[0]
        getitem_107: "f32[1, 2048, 1, 1]" = var_mean_52[1];  var_mean_52 = None
        add_276: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
        rsqrt_52: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
        sub_52: "f32[64, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, getitem_107)
        mul_364: "f32[64, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
        squeeze_156: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
        squeeze_157: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
        mul_365: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
        mul_366: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_318, 0.9)
        add_277: "f32[2048]" = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
        squeeze_158: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
        mul_367: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0003189792663476);  squeeze_158 = None
        mul_368: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_367, 0.1);  mul_367 = None
        mul_369: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_319, 0.9)
        add_278: "f32[2048]" = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
        unsqueeze_208: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_158, -1)
        unsqueeze_209: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        mul_370: "f32[64, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_209);  mul_364 = unsqueeze_209 = None
        unsqueeze_210: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_159, -1);  primals_159 = None
        unsqueeze_211: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        add_279: "f32[64, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_370, unsqueeze_211);  mul_370 = unsqueeze_211 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:166, code: out += residual
        add_280: "f32[64, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_279, relu_45);  add_279 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:173, code: out = self.relu(out)
        relu_48: "f32[64, 2048, 7, 7]" = torch.ops.aten.relu.default(add_280);  add_280 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:313, code: x = self.avgpool(x)
        mean: "f32[64, 2048, 1, 1]" = torch.ops.aten.mean.dim(relu_48, [-1, -2], True)
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:314, code: x = x.view(x.size(0), -1)
        view: "f32[64, 2048]" = torch.ops.aten.reshape.default(mean, [64, -1]);  mean = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:315, code: x = self.fc(x)
        permute: "f32[2048, 1000]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
        addmm: "f32[64, 1000]" = torch.ops.aten.addmm.default(primals_161, view, permute);  primals_161 = None
        permute_1: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:173, code: out = self.relu(out)
        le: "b8[64, 2048, 7, 7]" = torch.ops.aten.le.Scalar(relu_48, 0);  relu_48 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        unsqueeze_212: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
        unsqueeze_213: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 2);  unsqueeze_212 = None
        unsqueeze_214: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 3);  unsqueeze_213 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        unsqueeze_224: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
        unsqueeze_225: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 2);  unsqueeze_224 = None
        unsqueeze_226: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 3);  unsqueeze_225 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        unsqueeze_236: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
        unsqueeze_237: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 2);  unsqueeze_236 = None
        unsqueeze_238: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 3);  unsqueeze_237 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        unsqueeze_248: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
        unsqueeze_249: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 2);  unsqueeze_248 = None
        unsqueeze_250: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 3);  unsqueeze_249 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        unsqueeze_260: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
        unsqueeze_261: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
        unsqueeze_262: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        unsqueeze_272: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
        unsqueeze_273: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 2);  unsqueeze_272 = None
        unsqueeze_274: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 3);  unsqueeze_273 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:163, code: residual = self.downsample(x)
        unsqueeze_284: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
        unsqueeze_285: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
        unsqueeze_286: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        unsqueeze_296: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
        unsqueeze_297: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
        unsqueeze_298: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        unsqueeze_308: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
        unsqueeze_309: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
        unsqueeze_310: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        unsqueeze_320: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
        unsqueeze_321: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
        unsqueeze_322: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        unsqueeze_332: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
        unsqueeze_333: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 2);  unsqueeze_332 = None
        unsqueeze_334: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 3);  unsqueeze_333 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        unsqueeze_344: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
        unsqueeze_345: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 2);  unsqueeze_344 = None
        unsqueeze_346: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 3);  unsqueeze_345 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        unsqueeze_356: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
        unsqueeze_357: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 2);  unsqueeze_356 = None
        unsqueeze_358: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 3);  unsqueeze_357 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        unsqueeze_368: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
        unsqueeze_369: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 2);  unsqueeze_368 = None
        unsqueeze_370: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 3);  unsqueeze_369 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        unsqueeze_380: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
        unsqueeze_381: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 2);  unsqueeze_380 = None
        unsqueeze_382: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 3);  unsqueeze_381 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        unsqueeze_392: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
        unsqueeze_393: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 2);  unsqueeze_392 = None
        unsqueeze_394: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 3);  unsqueeze_393 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        unsqueeze_404: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
        unsqueeze_405: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
        unsqueeze_406: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        unsqueeze_416: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
        unsqueeze_417: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
        unsqueeze_418: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        unsqueeze_428: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
        unsqueeze_429: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
        unsqueeze_430: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        unsqueeze_440: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
        unsqueeze_441: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
        unsqueeze_442: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        unsqueeze_452: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
        unsqueeze_453: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
        unsqueeze_454: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        unsqueeze_464: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
        unsqueeze_465: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
        unsqueeze_466: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        unsqueeze_476: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
        unsqueeze_477: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
        unsqueeze_478: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        unsqueeze_488: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
        unsqueeze_489: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
        unsqueeze_490: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        unsqueeze_500: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_501: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
        unsqueeze_502: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:163, code: residual = self.downsample(x)
        unsqueeze_512: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
        unsqueeze_513: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 2);  unsqueeze_512 = None
        unsqueeze_514: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 3);  unsqueeze_513 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        unsqueeze_524: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
        unsqueeze_525: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 2);  unsqueeze_524 = None
        unsqueeze_526: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 3);  unsqueeze_525 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        unsqueeze_536: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
        unsqueeze_537: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 2);  unsqueeze_536 = None
        unsqueeze_538: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 3);  unsqueeze_537 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        unsqueeze_548: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_549: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 2);  unsqueeze_548 = None
        unsqueeze_550: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 3);  unsqueeze_549 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        unsqueeze_560: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
        unsqueeze_561: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 2);  unsqueeze_560 = None
        unsqueeze_562: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 3);  unsqueeze_561 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        unsqueeze_572: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_573: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
        unsqueeze_574: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        unsqueeze_584: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
        unsqueeze_585: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 2);  unsqueeze_584 = None
        unsqueeze_586: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 3);  unsqueeze_585 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        unsqueeze_596: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
        unsqueeze_597: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 2);  unsqueeze_596 = None
        unsqueeze_598: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 3);  unsqueeze_597 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        unsqueeze_608: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_609: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 2);  unsqueeze_608 = None
        unsqueeze_610: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 3);  unsqueeze_609 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        unsqueeze_620: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_621: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 2);  unsqueeze_620 = None
        unsqueeze_622: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 3);  unsqueeze_621 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        unsqueeze_632: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_633: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 2);  unsqueeze_632 = None
        unsqueeze_634: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 3);  unsqueeze_633 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        unsqueeze_644: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_645: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 2);  unsqueeze_644 = None
        unsqueeze_646: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 3);  unsqueeze_645 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        unsqueeze_656: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_657: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 2);  unsqueeze_656 = None
        unsqueeze_658: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 3);  unsqueeze_657 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:163, code: residual = self.downsample(x)
        unsqueeze_668: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_669: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 2);  unsqueeze_668 = None
        unsqueeze_670: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 3);  unsqueeze_669 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        unsqueeze_680: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_681: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 2);  unsqueeze_680 = None
        unsqueeze_682: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 3);  unsqueeze_681 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        unsqueeze_692: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_693: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 2);  unsqueeze_692 = None
        unsqueeze_694: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 3);  unsqueeze_693 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        unsqueeze_704: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_705: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 2);  unsqueeze_704 = None
        unsqueeze_706: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 3);  unsqueeze_705 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        unsqueeze_716: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_717: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 2);  unsqueeze_716 = None
        unsqueeze_718: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 3);  unsqueeze_717 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        unsqueeze_728: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_729: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 2);  unsqueeze_728 = None
        unsqueeze_730: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 3);  unsqueeze_729 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        unsqueeze_740: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_741: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 2);  unsqueeze_740 = None
        unsqueeze_742: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 3);  unsqueeze_741 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        unsqueeze_752: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_753: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 2);  unsqueeze_752 = None
        unsqueeze_754: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 3);  unsqueeze_753 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        unsqueeze_764: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_765: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 2);  unsqueeze_764 = None
        unsqueeze_766: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 3);  unsqueeze_765 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        unsqueeze_776: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_777: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 2);  unsqueeze_776 = None
        unsqueeze_778: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 3);  unsqueeze_777 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:163, code: residual = self.downsample(x)
        unsqueeze_788: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_789: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 2);  unsqueeze_788 = None
        unsqueeze_790: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 3);  unsqueeze_789 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:160, code: out = self.bn3(out)
        unsqueeze_800: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_801: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 2);  unsqueeze_800 = None
        unsqueeze_802: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 3);  unsqueeze_801 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:156, code: out = self.bn2(out)
        unsqueeze_812: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_813: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 2);  unsqueeze_812 = None
        unsqueeze_814: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 3);  unsqueeze_813 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:152, code: out = self.bn1(out)
        unsqueeze_824: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_825: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 2);  unsqueeze_824 = None
        unsqueeze_826: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 3);  unsqueeze_825 = None
        
        # File: /home/yuqixue2/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Classification/ConvNets/image_classification/models/resnet.py:307, code: x = self.bn1(x)
        unsqueeze_836: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_837: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 2);  unsqueeze_836 = None
        unsqueeze_838: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 3);  unsqueeze_837 = None
        
        # No stacktrace found for following nodes
        copy_: "f32[64]" = torch.ops.aten.copy_.default(primals_162, add_2);  primals_162 = add_2 = None
        copy__1: "f32[64]" = torch.ops.aten.copy_.default(primals_163, add_3);  primals_163 = add_3 = None
        copy__2: "i64[]" = torch.ops.aten.copy_.default(primals_164, add);  primals_164 = add = None
        copy__3: "f32[64]" = torch.ops.aten.copy_.default(primals_165, add_7);  primals_165 = add_7 = None
        copy__4: "f32[64]" = torch.ops.aten.copy_.default(primals_166, add_8);  primals_166 = add_8 = None
        copy__5: "i64[]" = torch.ops.aten.copy_.default(primals_167, add_5);  primals_167 = add_5 = None
        copy__6: "f32[64]" = torch.ops.aten.copy_.default(primals_168, add_12);  primals_168 = add_12 = None
        copy__7: "f32[64]" = torch.ops.aten.copy_.default(primals_169, add_13);  primals_169 = add_13 = None
        copy__8: "i64[]" = torch.ops.aten.copy_.default(primals_170, add_10);  primals_170 = add_10 = None
        copy__9: "f32[256]" = torch.ops.aten.copy_.default(primals_171, add_17);  primals_171 = add_17 = None
        copy__10: "f32[256]" = torch.ops.aten.copy_.default(primals_172, add_18);  primals_172 = add_18 = None
        copy__11: "i64[]" = torch.ops.aten.copy_.default(primals_173, add_15);  primals_173 = add_15 = None
        copy__12: "f32[256]" = torch.ops.aten.copy_.default(primals_174, add_22);  primals_174 = add_22 = None
        copy__13: "f32[256]" = torch.ops.aten.copy_.default(primals_175, add_23);  primals_175 = add_23 = None
        copy__14: "i64[]" = torch.ops.aten.copy_.default(primals_176, add_20);  primals_176 = add_20 = None
        copy__15: "f32[64]" = torch.ops.aten.copy_.default(primals_177, add_28);  primals_177 = add_28 = None
        copy__16: "f32[64]" = torch.ops.aten.copy_.default(primals_178, add_29);  primals_178 = add_29 = None
        copy__17: "i64[]" = torch.ops.aten.copy_.default(primals_179, add_26);  primals_179 = add_26 = None
        copy__18: "f32[64]" = torch.ops.aten.copy_.default(primals_180, add_33);  primals_180 = add_33 = None
        copy__19: "f32[64]" = torch.ops.aten.copy_.default(primals_181, add_34);  primals_181 = add_34 = None
        copy__20: "i64[]" = torch.ops.aten.copy_.default(primals_182, add_31);  primals_182 = add_31 = None
        copy__21: "f32[256]" = torch.ops.aten.copy_.default(primals_183, add_38);  primals_183 = add_38 = None
        copy__22: "f32[256]" = torch.ops.aten.copy_.default(primals_184, add_39);  primals_184 = add_39 = None
        copy__23: "i64[]" = torch.ops.aten.copy_.default(primals_185, add_36);  primals_185 = add_36 = None
        copy__24: "f32[64]" = torch.ops.aten.copy_.default(primals_186, add_44);  primals_186 = add_44 = None
        copy__25: "f32[64]" = torch.ops.aten.copy_.default(primals_187, add_45);  primals_187 = add_45 = None
        copy__26: "i64[]" = torch.ops.aten.copy_.default(primals_188, add_42);  primals_188 = add_42 = None
        copy__27: "f32[64]" = torch.ops.aten.copy_.default(primals_189, add_49);  primals_189 = add_49 = None
        copy__28: "f32[64]" = torch.ops.aten.copy_.default(primals_190, add_50);  primals_190 = add_50 = None
        copy__29: "i64[]" = torch.ops.aten.copy_.default(primals_191, add_47);  primals_191 = add_47 = None
        copy__30: "f32[256]" = torch.ops.aten.copy_.default(primals_192, add_54);  primals_192 = add_54 = None
        copy__31: "f32[256]" = torch.ops.aten.copy_.default(primals_193, add_55);  primals_193 = add_55 = None
        copy__32: "i64[]" = torch.ops.aten.copy_.default(primals_194, add_52);  primals_194 = add_52 = None
        copy__33: "f32[128]" = torch.ops.aten.copy_.default(primals_195, add_60);  primals_195 = add_60 = None
        copy__34: "f32[128]" = torch.ops.aten.copy_.default(primals_196, add_61);  primals_196 = add_61 = None
        copy__35: "i64[]" = torch.ops.aten.copy_.default(primals_197, add_58);  primals_197 = add_58 = None
        copy__36: "f32[128]" = torch.ops.aten.copy_.default(primals_198, add_65);  primals_198 = add_65 = None
        copy__37: "f32[128]" = torch.ops.aten.copy_.default(primals_199, add_66);  primals_199 = add_66 = None
        copy__38: "i64[]" = torch.ops.aten.copy_.default(primals_200, add_63);  primals_200 = add_63 = None
        copy__39: "f32[512]" = torch.ops.aten.copy_.default(primals_201, add_70);  primals_201 = add_70 = None
        copy__40: "f32[512]" = torch.ops.aten.copy_.default(primals_202, add_71);  primals_202 = add_71 = None
        copy__41: "i64[]" = torch.ops.aten.copy_.default(primals_203, add_68);  primals_203 = add_68 = None
        copy__42: "f32[512]" = torch.ops.aten.copy_.default(primals_204, add_75);  primals_204 = add_75 = None
        copy__43: "f32[512]" = torch.ops.aten.copy_.default(primals_205, add_76);  primals_205 = add_76 = None
        copy__44: "i64[]" = torch.ops.aten.copy_.default(primals_206, add_73);  primals_206 = add_73 = None
        copy__45: "f32[128]" = torch.ops.aten.copy_.default(primals_207, add_81);  primals_207 = add_81 = None
        copy__46: "f32[128]" = torch.ops.aten.copy_.default(primals_208, add_82);  primals_208 = add_82 = None
        copy__47: "i64[]" = torch.ops.aten.copy_.default(primals_209, add_79);  primals_209 = add_79 = None
        copy__48: "f32[128]" = torch.ops.aten.copy_.default(primals_210, add_86);  primals_210 = add_86 = None
        copy__49: "f32[128]" = torch.ops.aten.copy_.default(primals_211, add_87);  primals_211 = add_87 = None
        copy__50: "i64[]" = torch.ops.aten.copy_.default(primals_212, add_84);  primals_212 = add_84 = None
        copy__51: "f32[512]" = torch.ops.aten.copy_.default(primals_213, add_91);  primals_213 = add_91 = None
        copy__52: "f32[512]" = torch.ops.aten.copy_.default(primals_214, add_92);  primals_214 = add_92 = None
        copy__53: "i64[]" = torch.ops.aten.copy_.default(primals_215, add_89);  primals_215 = add_89 = None
        copy__54: "f32[128]" = torch.ops.aten.copy_.default(primals_216, add_97);  primals_216 = add_97 = None
        copy__55: "f32[128]" = torch.ops.aten.copy_.default(primals_217, add_98);  primals_217 = add_98 = None
        copy__56: "i64[]" = torch.ops.aten.copy_.default(primals_218, add_95);  primals_218 = add_95 = None
        copy__57: "f32[128]" = torch.ops.aten.copy_.default(primals_219, add_102);  primals_219 = add_102 = None
        copy__58: "f32[128]" = torch.ops.aten.copy_.default(primals_220, add_103);  primals_220 = add_103 = None
        copy__59: "i64[]" = torch.ops.aten.copy_.default(primals_221, add_100);  primals_221 = add_100 = None
        copy__60: "f32[512]" = torch.ops.aten.copy_.default(primals_222, add_107);  primals_222 = add_107 = None
        copy__61: "f32[512]" = torch.ops.aten.copy_.default(primals_223, add_108);  primals_223 = add_108 = None
        copy__62: "i64[]" = torch.ops.aten.copy_.default(primals_224, add_105);  primals_224 = add_105 = None
        copy__63: "f32[128]" = torch.ops.aten.copy_.default(primals_225, add_113);  primals_225 = add_113 = None
        copy__64: "f32[128]" = torch.ops.aten.copy_.default(primals_226, add_114);  primals_226 = add_114 = None
        copy__65: "i64[]" = torch.ops.aten.copy_.default(primals_227, add_111);  primals_227 = add_111 = None
        copy__66: "f32[128]" = torch.ops.aten.copy_.default(primals_228, add_118);  primals_228 = add_118 = None
        copy__67: "f32[128]" = torch.ops.aten.copy_.default(primals_229, add_119);  primals_229 = add_119 = None
        copy__68: "i64[]" = torch.ops.aten.copy_.default(primals_230, add_116);  primals_230 = add_116 = None
        copy__69: "f32[512]" = torch.ops.aten.copy_.default(primals_231, add_123);  primals_231 = add_123 = None
        copy__70: "f32[512]" = torch.ops.aten.copy_.default(primals_232, add_124);  primals_232 = add_124 = None
        copy__71: "i64[]" = torch.ops.aten.copy_.default(primals_233, add_121);  primals_233 = add_121 = None
        copy__72: "f32[256]" = torch.ops.aten.copy_.default(primals_234, add_129);  primals_234 = add_129 = None
        copy__73: "f32[256]" = torch.ops.aten.copy_.default(primals_235, add_130);  primals_235 = add_130 = None
        copy__74: "i64[]" = torch.ops.aten.copy_.default(primals_236, add_127);  primals_236 = add_127 = None
        copy__75: "f32[256]" = torch.ops.aten.copy_.default(primals_237, add_134);  primals_237 = add_134 = None
        copy__76: "f32[256]" = torch.ops.aten.copy_.default(primals_238, add_135);  primals_238 = add_135 = None
        copy__77: "i64[]" = torch.ops.aten.copy_.default(primals_239, add_132);  primals_239 = add_132 = None
        copy__78: "f32[1024]" = torch.ops.aten.copy_.default(primals_240, add_139);  primals_240 = add_139 = None
        copy__79: "f32[1024]" = torch.ops.aten.copy_.default(primals_241, add_140);  primals_241 = add_140 = None
        copy__80: "i64[]" = torch.ops.aten.copy_.default(primals_242, add_137);  primals_242 = add_137 = None
        copy__81: "f32[1024]" = torch.ops.aten.copy_.default(primals_243, add_144);  primals_243 = add_144 = None
        copy__82: "f32[1024]" = torch.ops.aten.copy_.default(primals_244, add_145);  primals_244 = add_145 = None
        copy__83: "i64[]" = torch.ops.aten.copy_.default(primals_245, add_142);  primals_245 = add_142 = None
        copy__84: "f32[256]" = torch.ops.aten.copy_.default(primals_246, add_150);  primals_246 = add_150 = None
        copy__85: "f32[256]" = torch.ops.aten.copy_.default(primals_247, add_151);  primals_247 = add_151 = None
        copy__86: "i64[]" = torch.ops.aten.copy_.default(primals_248, add_148);  primals_248 = add_148 = None
        copy__87: "f32[256]" = torch.ops.aten.copy_.default(primals_249, add_155);  primals_249 = add_155 = None
        copy__88: "f32[256]" = torch.ops.aten.copy_.default(primals_250, add_156);  primals_250 = add_156 = None
        copy__89: "i64[]" = torch.ops.aten.copy_.default(primals_251, add_153);  primals_251 = add_153 = None
        copy__90: "f32[1024]" = torch.ops.aten.copy_.default(primals_252, add_160);  primals_252 = add_160 = None
        copy__91: "f32[1024]" = torch.ops.aten.copy_.default(primals_253, add_161);  primals_253 = add_161 = None
        copy__92: "i64[]" = torch.ops.aten.copy_.default(primals_254, add_158);  primals_254 = add_158 = None
        copy__93: "f32[256]" = torch.ops.aten.copy_.default(primals_255, add_166);  primals_255 = add_166 = None
        copy__94: "f32[256]" = torch.ops.aten.copy_.default(primals_256, add_167);  primals_256 = add_167 = None
        copy__95: "i64[]" = torch.ops.aten.copy_.default(primals_257, add_164);  primals_257 = add_164 = None
        copy__96: "f32[256]" = torch.ops.aten.copy_.default(primals_258, add_171);  primals_258 = add_171 = None
        copy__97: "f32[256]" = torch.ops.aten.copy_.default(primals_259, add_172);  primals_259 = add_172 = None
        copy__98: "i64[]" = torch.ops.aten.copy_.default(primals_260, add_169);  primals_260 = add_169 = None
        copy__99: "f32[1024]" = torch.ops.aten.copy_.default(primals_261, add_176);  primals_261 = add_176 = None
        copy__100: "f32[1024]" = torch.ops.aten.copy_.default(primals_262, add_177);  primals_262 = add_177 = None
        copy__101: "i64[]" = torch.ops.aten.copy_.default(primals_263, add_174);  primals_263 = add_174 = None
        copy__102: "f32[256]" = torch.ops.aten.copy_.default(primals_264, add_182);  primals_264 = add_182 = None
        copy__103: "f32[256]" = torch.ops.aten.copy_.default(primals_265, add_183);  primals_265 = add_183 = None
        copy__104: "i64[]" = torch.ops.aten.copy_.default(primals_266, add_180);  primals_266 = add_180 = None
        copy__105: "f32[256]" = torch.ops.aten.copy_.default(primals_267, add_187);  primals_267 = add_187 = None
        copy__106: "f32[256]" = torch.ops.aten.copy_.default(primals_268, add_188);  primals_268 = add_188 = None
        copy__107: "i64[]" = torch.ops.aten.copy_.default(primals_269, add_185);  primals_269 = add_185 = None
        copy__108: "f32[1024]" = torch.ops.aten.copy_.default(primals_270, add_192);  primals_270 = add_192 = None
        copy__109: "f32[1024]" = torch.ops.aten.copy_.default(primals_271, add_193);  primals_271 = add_193 = None
        copy__110: "i64[]" = torch.ops.aten.copy_.default(primals_272, add_190);  primals_272 = add_190 = None
        copy__111: "f32[256]" = torch.ops.aten.copy_.default(primals_273, add_198);  primals_273 = add_198 = None
        copy__112: "f32[256]" = torch.ops.aten.copy_.default(primals_274, add_199);  primals_274 = add_199 = None
        copy__113: "i64[]" = torch.ops.aten.copy_.default(primals_275, add_196);  primals_275 = add_196 = None
        copy__114: "f32[256]" = torch.ops.aten.copy_.default(primals_276, add_203);  primals_276 = add_203 = None
        copy__115: "f32[256]" = torch.ops.aten.copy_.default(primals_277, add_204);  primals_277 = add_204 = None
        copy__116: "i64[]" = torch.ops.aten.copy_.default(primals_278, add_201);  primals_278 = add_201 = None
        copy__117: "f32[1024]" = torch.ops.aten.copy_.default(primals_279, add_208);  primals_279 = add_208 = None
        copy__118: "f32[1024]" = torch.ops.aten.copy_.default(primals_280, add_209);  primals_280 = add_209 = None
        copy__119: "i64[]" = torch.ops.aten.copy_.default(primals_281, add_206);  primals_281 = add_206 = None
        copy__120: "f32[256]" = torch.ops.aten.copy_.default(primals_282, add_214);  primals_282 = add_214 = None
        copy__121: "f32[256]" = torch.ops.aten.copy_.default(primals_283, add_215);  primals_283 = add_215 = None
        copy__122: "i64[]" = torch.ops.aten.copy_.default(primals_284, add_212);  primals_284 = add_212 = None
        copy__123: "f32[256]" = torch.ops.aten.copy_.default(primals_285, add_219);  primals_285 = add_219 = None
        copy__124: "f32[256]" = torch.ops.aten.copy_.default(primals_286, add_220);  primals_286 = add_220 = None
        copy__125: "i64[]" = torch.ops.aten.copy_.default(primals_287, add_217);  primals_287 = add_217 = None
        copy__126: "f32[1024]" = torch.ops.aten.copy_.default(primals_288, add_224);  primals_288 = add_224 = None
        copy__127: "f32[1024]" = torch.ops.aten.copy_.default(primals_289, add_225);  primals_289 = add_225 = None
        copy__128: "i64[]" = torch.ops.aten.copy_.default(primals_290, add_222);  primals_290 = add_222 = None
        copy__129: "f32[512]" = torch.ops.aten.copy_.default(primals_291, add_230);  primals_291 = add_230 = None
        copy__130: "f32[512]" = torch.ops.aten.copy_.default(primals_292, add_231);  primals_292 = add_231 = None
        copy__131: "i64[]" = torch.ops.aten.copy_.default(primals_293, add_228);  primals_293 = add_228 = None
        copy__132: "f32[512]" = torch.ops.aten.copy_.default(primals_294, add_235);  primals_294 = add_235 = None
        copy__133: "f32[512]" = torch.ops.aten.copy_.default(primals_295, add_236);  primals_295 = add_236 = None
        copy__134: "i64[]" = torch.ops.aten.copy_.default(primals_296, add_233);  primals_296 = add_233 = None
        copy__135: "f32[2048]" = torch.ops.aten.copy_.default(primals_297, add_240);  primals_297 = add_240 = None
        copy__136: "f32[2048]" = torch.ops.aten.copy_.default(primals_298, add_241);  primals_298 = add_241 = None
        copy__137: "i64[]" = torch.ops.aten.copy_.default(primals_299, add_238);  primals_299 = add_238 = None
        copy__138: "f32[2048]" = torch.ops.aten.copy_.default(primals_300, add_245);  primals_300 = add_245 = None
        copy__139: "f32[2048]" = torch.ops.aten.copy_.default(primals_301, add_246);  primals_301 = add_246 = None
        copy__140: "i64[]" = torch.ops.aten.copy_.default(primals_302, add_243);  primals_302 = add_243 = None
        copy__141: "f32[512]" = torch.ops.aten.copy_.default(primals_303, add_251);  primals_303 = add_251 = None
        copy__142: "f32[512]" = torch.ops.aten.copy_.default(primals_304, add_252);  primals_304 = add_252 = None
        copy__143: "i64[]" = torch.ops.aten.copy_.default(primals_305, add_249);  primals_305 = add_249 = None
        copy__144: "f32[512]" = torch.ops.aten.copy_.default(primals_306, add_256);  primals_306 = add_256 = None
        copy__145: "f32[512]" = torch.ops.aten.copy_.default(primals_307, add_257);  primals_307 = add_257 = None
        copy__146: "i64[]" = torch.ops.aten.copy_.default(primals_308, add_254);  primals_308 = add_254 = None
        copy__147: "f32[2048]" = torch.ops.aten.copy_.default(primals_309, add_261);  primals_309 = add_261 = None
        copy__148: "f32[2048]" = torch.ops.aten.copy_.default(primals_310, add_262);  primals_310 = add_262 = None
        copy__149: "i64[]" = torch.ops.aten.copy_.default(primals_311, add_259);  primals_311 = add_259 = None
        copy__150: "f32[512]" = torch.ops.aten.copy_.default(primals_312, add_267);  primals_312 = add_267 = None
        copy__151: "f32[512]" = torch.ops.aten.copy_.default(primals_313, add_268);  primals_313 = add_268 = None
        copy__152: "i64[]" = torch.ops.aten.copy_.default(primals_314, add_265);  primals_314 = add_265 = None
        copy__153: "f32[512]" = torch.ops.aten.copy_.default(primals_315, add_272);  primals_315 = add_272 = None
        copy__154: "f32[512]" = torch.ops.aten.copy_.default(primals_316, add_273);  primals_316 = add_273 = None
        copy__155: "i64[]" = torch.ops.aten.copy_.default(primals_317, add_270);  primals_317 = add_270 = None
        copy__156: "f32[2048]" = torch.ops.aten.copy_.default(primals_318, add_277);  primals_318 = add_277 = None
        copy__157: "f32[2048]" = torch.ops.aten.copy_.default(primals_319, add_278);  primals_319 = add_278 = None
        copy__158: "i64[]" = torch.ops.aten.copy_.default(primals_320, add_275);  primals_320 = add_275 = None
        return [addmm, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_321, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, relu_4, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, relu_8, convolution_10, squeeze_31, relu_9, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, relu_11, convolution_13, squeeze_40, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, relu_14, convolution_17, squeeze_52, relu_15, convolution_18, squeeze_55, relu_16, convolution_19, squeeze_58, relu_17, convolution_20, squeeze_61, relu_18, convolution_21, squeeze_64, relu_19, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, relu_21, convolution_24, squeeze_73, relu_22, convolution_25, squeeze_76, relu_23, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, relu_27, convolution_31, squeeze_94, relu_28, convolution_32, squeeze_97, relu_29, convolution_33, squeeze_100, relu_30, convolution_34, squeeze_103, relu_31, convolution_35, squeeze_106, relu_32, convolution_36, squeeze_109, relu_33, convolution_37, squeeze_112, relu_34, convolution_38, squeeze_115, relu_35, convolution_39, squeeze_118, relu_36, convolution_40, squeeze_121, relu_37, convolution_41, squeeze_124, relu_38, convolution_42, squeeze_127, relu_39, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, relu_41, convolution_45, squeeze_136, convolution_46, squeeze_139, relu_42, convolution_47, squeeze_142, relu_43, convolution_48, squeeze_145, relu_44, convolution_49, squeeze_148, relu_45, convolution_50, squeeze_151, relu_46, convolution_51, squeeze_154, relu_47, convolution_52, squeeze_157, view, permute_1, le, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838]
        