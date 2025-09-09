class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 3, 3, 3]", primals_2: "f32[32]", primals_4: "f32[32, 32, 3, 3]", primals_5: "f32[32]", primals_7: "f32[64, 32, 3, 3]", primals_8: "f32[64]", primals_10: "f32[80, 64, 1, 1]", primals_11: "f32[80]", primals_13: "f32[192, 80, 3, 3]", primals_14: "f32[192]", primals_16: "f32[64, 192, 1, 1]", primals_17: "f32[64]", primals_19: "f32[48, 192, 1, 1]", primals_20: "f32[48]", primals_22: "f32[64, 48, 5, 5]", primals_23: "f32[64]", primals_25: "f32[64, 192, 1, 1]", primals_26: "f32[64]", primals_28: "f32[96, 64, 3, 3]", primals_29: "f32[96]", primals_31: "f32[96, 96, 3, 3]", primals_32: "f32[96]", primals_34: "f32[32, 192, 1, 1]", primals_35: "f32[32]", primals_37: "f32[64, 256, 1, 1]", primals_38: "f32[64]", primals_40: "f32[48, 256, 1, 1]", primals_41: "f32[48]", primals_43: "f32[64, 48, 5, 5]", primals_44: "f32[64]", primals_46: "f32[64, 256, 1, 1]", primals_47: "f32[64]", primals_49: "f32[96, 64, 3, 3]", primals_50: "f32[96]", primals_52: "f32[96, 96, 3, 3]", primals_53: "f32[96]", primals_55: "f32[64, 256, 1, 1]", primals_56: "f32[64]", primals_58: "f32[64, 288, 1, 1]", primals_59: "f32[64]", primals_61: "f32[48, 288, 1, 1]", primals_62: "f32[48]", primals_64: "f32[64, 48, 5, 5]", primals_65: "f32[64]", primals_67: "f32[64, 288, 1, 1]", primals_68: "f32[64]", primals_70: "f32[96, 64, 3, 3]", primals_71: "f32[96]", primals_73: "f32[96, 96, 3, 3]", primals_74: "f32[96]", primals_76: "f32[64, 288, 1, 1]", primals_77: "f32[64]", primals_79: "f32[384, 288, 3, 3]", primals_80: "f32[384]", primals_82: "f32[64, 288, 1, 1]", primals_83: "f32[64]", primals_85: "f32[96, 64, 3, 3]", primals_86: "f32[96]", primals_88: "f32[96, 96, 3, 3]", primals_89: "f32[96]", primals_91: "f32[192, 768, 1, 1]", primals_92: "f32[192]", primals_94: "f32[128, 768, 1, 1]", primals_95: "f32[128]", primals_97: "f32[128, 128, 1, 7]", primals_98: "f32[128]", primals_100: "f32[192, 128, 7, 1]", primals_101: "f32[192]", primals_103: "f32[128, 768, 1, 1]", primals_104: "f32[128]", primals_106: "f32[128, 128, 7, 1]", primals_107: "f32[128]", primals_109: "f32[128, 128, 1, 7]", primals_110: "f32[128]", primals_112: "f32[128, 128, 7, 1]", primals_113: "f32[128]", primals_115: "f32[192, 128, 1, 7]", primals_116: "f32[192]", primals_118: "f32[192, 768, 1, 1]", primals_119: "f32[192]", primals_121: "f32[192, 768, 1, 1]", primals_122: "f32[192]", primals_124: "f32[160, 768, 1, 1]", primals_125: "f32[160]", primals_127: "f32[160, 160, 1, 7]", primals_128: "f32[160]", primals_130: "f32[192, 160, 7, 1]", primals_131: "f32[192]", primals_133: "f32[160, 768, 1, 1]", primals_134: "f32[160]", primals_136: "f32[160, 160, 7, 1]", primals_137: "f32[160]", primals_139: "f32[160, 160, 1, 7]", primals_140: "f32[160]", primals_142: "f32[160, 160, 7, 1]", primals_143: "f32[160]", primals_145: "f32[192, 160, 1, 7]", primals_146: "f32[192]", primals_148: "f32[192, 768, 1, 1]", primals_149: "f32[192]", primals_151: "f32[192, 768, 1, 1]", primals_152: "f32[192]", primals_154: "f32[160, 768, 1, 1]", primals_155: "f32[160]", primals_157: "f32[160, 160, 1, 7]", primals_158: "f32[160]", primals_160: "f32[192, 160, 7, 1]", primals_161: "f32[192]", primals_163: "f32[160, 768, 1, 1]", primals_164: "f32[160]", primals_166: "f32[160, 160, 7, 1]", primals_167: "f32[160]", primals_169: "f32[160, 160, 1, 7]", primals_170: "f32[160]", primals_172: "f32[160, 160, 7, 1]", primals_173: "f32[160]", primals_175: "f32[192, 160, 1, 7]", primals_176: "f32[192]", primals_178: "f32[192, 768, 1, 1]", primals_179: "f32[192]", primals_181: "f32[192, 768, 1, 1]", primals_182: "f32[192]", primals_184: "f32[192, 768, 1, 1]", primals_185: "f32[192]", primals_187: "f32[192, 192, 1, 7]", primals_188: "f32[192]", primals_190: "f32[192, 192, 7, 1]", primals_191: "f32[192]", primals_193: "f32[192, 768, 1, 1]", primals_194: "f32[192]", primals_196: "f32[192, 192, 7, 1]", primals_197: "f32[192]", primals_199: "f32[192, 192, 1, 7]", primals_200: "f32[192]", primals_202: "f32[192, 192, 7, 1]", primals_203: "f32[192]", primals_205: "f32[192, 192, 1, 7]", primals_206: "f32[192]", primals_208: "f32[192, 768, 1, 1]", primals_209: "f32[192]", primals_211: "f32[128, 768, 1, 1]", primals_212: "f32[128]", primals_214: "f32[768, 128, 5, 5]", primals_215: "f32[768]", primals_219: "f32[192, 768, 1, 1]", primals_220: "f32[192]", primals_222: "f32[320, 192, 3, 3]", primals_223: "f32[320]", primals_225: "f32[192, 768, 1, 1]", primals_226: "f32[192]", primals_228: "f32[192, 192, 1, 7]", primals_229: "f32[192]", primals_231: "f32[192, 192, 7, 1]", primals_232: "f32[192]", primals_234: "f32[192, 192, 3, 3]", primals_235: "f32[192]", primals_237: "f32[320, 1280, 1, 1]", primals_238: "f32[320]", primals_240: "f32[384, 1280, 1, 1]", primals_241: "f32[384]", primals_243: "f32[384, 384, 1, 3]", primals_244: "f32[384]", primals_246: "f32[384, 384, 3, 1]", primals_247: "f32[384]", primals_249: "f32[448, 1280, 1, 1]", primals_250: "f32[448]", primals_252: "f32[384, 448, 3, 3]", primals_253: "f32[384]", primals_255: "f32[384, 384, 1, 3]", primals_256: "f32[384]", primals_258: "f32[384, 384, 3, 1]", primals_259: "f32[384]", primals_261: "f32[192, 1280, 1, 1]", primals_262: "f32[192]", primals_264: "f32[320, 2048, 1, 1]", primals_265: "f32[320]", primals_267: "f32[384, 2048, 1, 1]", primals_268: "f32[384]", primals_270: "f32[384, 384, 1, 3]", primals_271: "f32[384]", primals_273: "f32[384, 384, 3, 1]", primals_274: "f32[384]", primals_276: "f32[448, 2048, 1, 1]", primals_277: "f32[448]", primals_279: "f32[384, 448, 3, 3]", primals_280: "f32[384]", primals_282: "f32[384, 384, 1, 3]", primals_283: "f32[384]", primals_285: "f32[384, 384, 3, 1]", primals_286: "f32[384]", primals_288: "f32[192, 2048, 1, 1]", primals_289: "f32[192]", cat: "f32[32, 3, 299, 299]", convolution: "f32[32, 32, 149, 149]", squeeze_1: "f32[32]", relu: "f32[32, 32, 149, 149]", convolution_1: "f32[32, 32, 147, 147]", squeeze_4: "f32[32]", relu_1: "f32[32, 32, 147, 147]", convolution_2: "f32[32, 64, 147, 147]", squeeze_7: "f32[64]", relu_2: "f32[32, 64, 147, 147]", getitem_6: "f32[32, 64, 73, 73]", getitem_7: "i64[32, 64, 73, 73]", convolution_3: "f32[32, 80, 73, 73]", squeeze_10: "f32[80]", relu_3: "f32[32, 80, 73, 73]", convolution_4: "f32[32, 192, 71, 71]", squeeze_13: "f32[192]", relu_4: "f32[32, 192, 71, 71]", getitem_12: "f32[32, 192, 35, 35]", getitem_13: "i64[32, 192, 35, 35]", convolution_5: "f32[32, 64, 35, 35]", squeeze_16: "f32[64]", convolution_6: "f32[32, 48, 35, 35]", squeeze_19: "f32[48]", relu_6: "f32[32, 48, 35, 35]", convolution_7: "f32[32, 64, 35, 35]", squeeze_22: "f32[64]", convolution_8: "f32[32, 64, 35, 35]", squeeze_25: "f32[64]", relu_8: "f32[32, 64, 35, 35]", convolution_9: "f32[32, 96, 35, 35]", squeeze_28: "f32[96]", relu_9: "f32[32, 96, 35, 35]", convolution_10: "f32[32, 96, 35, 35]", squeeze_31: "f32[96]", avg_pool2d: "f32[32, 192, 35, 35]", convolution_11: "f32[32, 32, 35, 35]", squeeze_34: "f32[32]", cat_1: "f32[32, 256, 35, 35]", convolution_12: "f32[32, 64, 35, 35]", squeeze_37: "f32[64]", convolution_13: "f32[32, 48, 35, 35]", squeeze_40: "f32[48]", relu_13: "f32[32, 48, 35, 35]", convolution_14: "f32[32, 64, 35, 35]", squeeze_43: "f32[64]", convolution_15: "f32[32, 64, 35, 35]", squeeze_46: "f32[64]", relu_15: "f32[32, 64, 35, 35]", convolution_16: "f32[32, 96, 35, 35]", squeeze_49: "f32[96]", relu_16: "f32[32, 96, 35, 35]", convolution_17: "f32[32, 96, 35, 35]", squeeze_52: "f32[96]", avg_pool2d_1: "f32[32, 256, 35, 35]", convolution_18: "f32[32, 64, 35, 35]", squeeze_55: "f32[64]", cat_2: "f32[32, 288, 35, 35]", convolution_19: "f32[32, 64, 35, 35]", squeeze_58: "f32[64]", convolution_20: "f32[32, 48, 35, 35]", squeeze_61: "f32[48]", relu_20: "f32[32, 48, 35, 35]", convolution_21: "f32[32, 64, 35, 35]", squeeze_64: "f32[64]", convolution_22: "f32[32, 64, 35, 35]", squeeze_67: "f32[64]", relu_22: "f32[32, 64, 35, 35]", convolution_23: "f32[32, 96, 35, 35]", squeeze_70: "f32[96]", relu_23: "f32[32, 96, 35, 35]", convolution_24: "f32[32, 96, 35, 35]", squeeze_73: "f32[96]", avg_pool2d_2: "f32[32, 288, 35, 35]", convolution_25: "f32[32, 64, 35, 35]", squeeze_76: "f32[64]", cat_3: "f32[32, 288, 35, 35]", convolution_26: "f32[32, 384, 17, 17]", squeeze_79: "f32[384]", convolution_27: "f32[32, 64, 35, 35]", squeeze_82: "f32[64]", relu_27: "f32[32, 64, 35, 35]", convolution_28: "f32[32, 96, 35, 35]", squeeze_85: "f32[96]", relu_28: "f32[32, 96, 35, 35]", convolution_29: "f32[32, 96, 17, 17]", squeeze_88: "f32[96]", getitem_65: "i64[32, 288, 17, 17]", cat_4: "f32[32, 768, 17, 17]", convolution_30: "f32[32, 192, 17, 17]", squeeze_91: "f32[192]", convolution_31: "f32[32, 128, 17, 17]", squeeze_94: "f32[128]", relu_31: "f32[32, 128, 17, 17]", convolution_32: "f32[32, 128, 17, 17]", squeeze_97: "f32[128]", relu_32: "f32[32, 128, 17, 17]", convolution_33: "f32[32, 192, 17, 17]", squeeze_100: "f32[192]", convolution_34: "f32[32, 128, 17, 17]", squeeze_103: "f32[128]", relu_34: "f32[32, 128, 17, 17]", convolution_35: "f32[32, 128, 17, 17]", squeeze_106: "f32[128]", relu_35: "f32[32, 128, 17, 17]", convolution_36: "f32[32, 128, 17, 17]", squeeze_109: "f32[128]", relu_36: "f32[32, 128, 17, 17]", convolution_37: "f32[32, 128, 17, 17]", squeeze_112: "f32[128]", relu_37: "f32[32, 128, 17, 17]", convolution_38: "f32[32, 192, 17, 17]", squeeze_115: "f32[192]", avg_pool2d_3: "f32[32, 768, 17, 17]", convolution_39: "f32[32, 192, 17, 17]", squeeze_118: "f32[192]", cat_5: "f32[32, 768, 17, 17]", convolution_40: "f32[32, 192, 17, 17]", squeeze_121: "f32[192]", convolution_41: "f32[32, 160, 17, 17]", squeeze_124: "f32[160]", relu_41: "f32[32, 160, 17, 17]", convolution_42: "f32[32, 160, 17, 17]", squeeze_127: "f32[160]", relu_42: "f32[32, 160, 17, 17]", convolution_43: "f32[32, 192, 17, 17]", squeeze_130: "f32[192]", convolution_44: "f32[32, 160, 17, 17]", squeeze_133: "f32[160]", relu_44: "f32[32, 160, 17, 17]", convolution_45: "f32[32, 160, 17, 17]", squeeze_136: "f32[160]", relu_45: "f32[32, 160, 17, 17]", convolution_46: "f32[32, 160, 17, 17]", squeeze_139: "f32[160]", relu_46: "f32[32, 160, 17, 17]", convolution_47: "f32[32, 160, 17, 17]", squeeze_142: "f32[160]", relu_47: "f32[32, 160, 17, 17]", convolution_48: "f32[32, 192, 17, 17]", squeeze_145: "f32[192]", avg_pool2d_4: "f32[32, 768, 17, 17]", convolution_49: "f32[32, 192, 17, 17]", squeeze_148: "f32[192]", cat_6: "f32[32, 768, 17, 17]", convolution_50: "f32[32, 192, 17, 17]", squeeze_151: "f32[192]", convolution_51: "f32[32, 160, 17, 17]", squeeze_154: "f32[160]", relu_51: "f32[32, 160, 17, 17]", convolution_52: "f32[32, 160, 17, 17]", squeeze_157: "f32[160]", relu_52: "f32[32, 160, 17, 17]", convolution_53: "f32[32, 192, 17, 17]", squeeze_160: "f32[192]", convolution_54: "f32[32, 160, 17, 17]", squeeze_163: "f32[160]", relu_54: "f32[32, 160, 17, 17]", convolution_55: "f32[32, 160, 17, 17]", squeeze_166: "f32[160]", relu_55: "f32[32, 160, 17, 17]", convolution_56: "f32[32, 160, 17, 17]", squeeze_169: "f32[160]", relu_56: "f32[32, 160, 17, 17]", convolution_57: "f32[32, 160, 17, 17]", squeeze_172: "f32[160]", relu_57: "f32[32, 160, 17, 17]", convolution_58: "f32[32, 192, 17, 17]", squeeze_175: "f32[192]", avg_pool2d_5: "f32[32, 768, 17, 17]", convolution_59: "f32[32, 192, 17, 17]", squeeze_178: "f32[192]", cat_7: "f32[32, 768, 17, 17]", convolution_60: "f32[32, 192, 17, 17]", squeeze_181: "f32[192]", convolution_61: "f32[32, 192, 17, 17]", squeeze_184: "f32[192]", relu_61: "f32[32, 192, 17, 17]", convolution_62: "f32[32, 192, 17, 17]", squeeze_187: "f32[192]", relu_62: "f32[32, 192, 17, 17]", convolution_63: "f32[32, 192, 17, 17]", squeeze_190: "f32[192]", convolution_64: "f32[32, 192, 17, 17]", squeeze_193: "f32[192]", relu_64: "f32[32, 192, 17, 17]", convolution_65: "f32[32, 192, 17, 17]", squeeze_196: "f32[192]", relu_65: "f32[32, 192, 17, 17]", convolution_66: "f32[32, 192, 17, 17]", squeeze_199: "f32[192]", relu_66: "f32[32, 192, 17, 17]", convolution_67: "f32[32, 192, 17, 17]", squeeze_202: "f32[192]", relu_67: "f32[32, 192, 17, 17]", convolution_68: "f32[32, 192, 17, 17]", squeeze_205: "f32[192]", avg_pool2d_6: "f32[32, 768, 17, 17]", convolution_69: "f32[32, 192, 17, 17]", squeeze_208: "f32[192]", cat_8: "f32[32, 768, 17, 17]", avg_pool2d_7: "f32[32, 768, 5, 5]", convolution_70: "f32[32, 128, 5, 5]", squeeze_211: "f32[128]", relu_70: "f32[32, 128, 5, 5]", convolution_71: "f32[32, 768, 1, 1]", squeeze_214: "f32[768]", view: "f32[32, 768]", convolution_72: "f32[32, 192, 17, 17]", squeeze_217: "f32[192]", relu_72: "f32[32, 192, 17, 17]", convolution_73: "f32[32, 320, 8, 8]", squeeze_220: "f32[320]", convolution_74: "f32[32, 192, 17, 17]", squeeze_223: "f32[192]", relu_74: "f32[32, 192, 17, 17]", convolution_75: "f32[32, 192, 17, 17]", squeeze_226: "f32[192]", relu_75: "f32[32, 192, 17, 17]", convolution_76: "f32[32, 192, 17, 17]", squeeze_229: "f32[192]", relu_76: "f32[32, 192, 17, 17]", convolution_77: "f32[32, 192, 8, 8]", squeeze_232: "f32[192]", getitem_163: "i64[32, 768, 8, 8]", cat_9: "f32[32, 1280, 8, 8]", convolution_78: "f32[32, 320, 8, 8]", squeeze_235: "f32[320]", convolution_79: "f32[32, 384, 8, 8]", squeeze_238: "f32[384]", relu_79: "f32[32, 384, 8, 8]", convolution_80: "f32[32, 384, 8, 8]", squeeze_241: "f32[384]", convolution_81: "f32[32, 384, 8, 8]", squeeze_244: "f32[384]", convolution_82: "f32[32, 448, 8, 8]", squeeze_247: "f32[448]", relu_82: "f32[32, 448, 8, 8]", convolution_83: "f32[32, 384, 8, 8]", squeeze_250: "f32[384]", relu_83: "f32[32, 384, 8, 8]", convolution_84: "f32[32, 384, 8, 8]", squeeze_253: "f32[384]", convolution_85: "f32[32, 384, 8, 8]", squeeze_256: "f32[384]", avg_pool2d_8: "f32[32, 1280, 8, 8]", convolution_86: "f32[32, 192, 8, 8]", squeeze_259: "f32[192]", cat_12: "f32[32, 2048, 8, 8]", convolution_87: "f32[32, 320, 8, 8]", squeeze_262: "f32[320]", convolution_88: "f32[32, 384, 8, 8]", squeeze_265: "f32[384]", relu_88: "f32[32, 384, 8, 8]", convolution_89: "f32[32, 384, 8, 8]", squeeze_268: "f32[384]", convolution_90: "f32[32, 384, 8, 8]", squeeze_271: "f32[384]", convolution_91: "f32[32, 448, 8, 8]", squeeze_274: "f32[448]", relu_91: "f32[32, 448, 8, 8]", convolution_92: "f32[32, 384, 8, 8]", squeeze_277: "f32[384]", relu_92: "f32[32, 384, 8, 8]", convolution_93: "f32[32, 384, 8, 8]", squeeze_280: "f32[384]", convolution_94: "f32[32, 384, 8, 8]", squeeze_283: "f32[384]", avg_pool2d_9: "f32[32, 2048, 8, 8]", convolution_95: "f32[32, 192, 8, 8]", squeeze_286: "f32[192]", gt: "b8[32, 2048, 1, 1]", view_1: "f32[32, 2048]", permute_2: "f32[1000, 2048]", le: "b8[32, 192, 8, 8]", unsqueeze_389: "f32[1, 192, 1, 1]", le_1: "b8[32, 384, 8, 8]", unsqueeze_401: "f32[1, 384, 1, 1]", le_2: "b8[32, 384, 8, 8]", unsqueeze_413: "f32[1, 384, 1, 1]", unsqueeze_425: "f32[1, 384, 1, 1]", unsqueeze_437: "f32[1, 448, 1, 1]", le_5: "b8[32, 384, 8, 8]", unsqueeze_449: "f32[1, 384, 1, 1]", le_6: "b8[32, 384, 8, 8]", unsqueeze_461: "f32[1, 384, 1, 1]", unsqueeze_473: "f32[1, 384, 1, 1]", le_8: "b8[32, 320, 8, 8]", unsqueeze_485: "f32[1, 320, 1, 1]", le_9: "b8[32, 192, 8, 8]", unsqueeze_497: "f32[1, 192, 1, 1]", le_10: "b8[32, 384, 8, 8]", unsqueeze_509: "f32[1, 384, 1, 1]", le_11: "b8[32, 384, 8, 8]", unsqueeze_521: "f32[1, 384, 1, 1]", unsqueeze_533: "f32[1, 384, 1, 1]", unsqueeze_545: "f32[1, 448, 1, 1]", le_14: "b8[32, 384, 8, 8]", unsqueeze_557: "f32[1, 384, 1, 1]", le_15: "b8[32, 384, 8, 8]", unsqueeze_569: "f32[1, 384, 1, 1]", unsqueeze_581: "f32[1, 384, 1, 1]", le_17: "b8[32, 320, 8, 8]", unsqueeze_593: "f32[1, 320, 1, 1]", le_18: "b8[32, 192, 8, 8]", unsqueeze_605: "f32[1, 192, 1, 1]", unsqueeze_617: "f32[1, 192, 1, 1]", unsqueeze_629: "f32[1, 192, 1, 1]", unsqueeze_641: "f32[1, 192, 1, 1]", le_22: "b8[32, 320, 8, 8]", unsqueeze_653: "f32[1, 320, 1, 1]", unsqueeze_665: "f32[1, 192, 1, 1]", permute_6: "f32[1000, 768]", le_24: "b8[32, 768, 1, 1]", unsqueeze_677: "f32[1, 768, 1, 1]", unsqueeze_689: "f32[1, 128, 1, 1]", le_26: "b8[32, 192, 17, 17]", unsqueeze_701: "f32[1, 192, 1, 1]", le_27: "b8[32, 192, 17, 17]", unsqueeze_713: "f32[1, 192, 1, 1]", unsqueeze_725: "f32[1, 192, 1, 1]", unsqueeze_737: "f32[1, 192, 1, 1]", unsqueeze_749: "f32[1, 192, 1, 1]", unsqueeze_761: "f32[1, 192, 1, 1]", le_32: "b8[32, 192, 17, 17]", unsqueeze_773: "f32[1, 192, 1, 1]", unsqueeze_785: "f32[1, 192, 1, 1]", unsqueeze_797: "f32[1, 192, 1, 1]", le_35: "b8[32, 192, 17, 17]", unsqueeze_809: "f32[1, 192, 1, 1]", le_36: "b8[32, 192, 17, 17]", unsqueeze_821: "f32[1, 192, 1, 1]", le_37: "b8[32, 192, 17, 17]", unsqueeze_833: "f32[1, 192, 1, 1]", unsqueeze_845: "f32[1, 160, 1, 1]", unsqueeze_857: "f32[1, 160, 1, 1]", unsqueeze_869: "f32[1, 160, 1, 1]", unsqueeze_881: "f32[1, 160, 1, 1]", le_42: "b8[32, 192, 17, 17]", unsqueeze_893: "f32[1, 192, 1, 1]", unsqueeze_905: "f32[1, 160, 1, 1]", unsqueeze_917: "f32[1, 160, 1, 1]", le_45: "b8[32, 192, 17, 17]", unsqueeze_929: "f32[1, 192, 1, 1]", le_46: "b8[32, 192, 17, 17]", unsqueeze_941: "f32[1, 192, 1, 1]", le_47: "b8[32, 192, 17, 17]", unsqueeze_953: "f32[1, 192, 1, 1]", unsqueeze_965: "f32[1, 160, 1, 1]", unsqueeze_977: "f32[1, 160, 1, 1]", unsqueeze_989: "f32[1, 160, 1, 1]", unsqueeze_1001: "f32[1, 160, 1, 1]", le_52: "b8[32, 192, 17, 17]", unsqueeze_1013: "f32[1, 192, 1, 1]", unsqueeze_1025: "f32[1, 160, 1, 1]", unsqueeze_1037: "f32[1, 160, 1, 1]", le_55: "b8[32, 192, 17, 17]", unsqueeze_1049: "f32[1, 192, 1, 1]", le_56: "b8[32, 192, 17, 17]", unsqueeze_1061: "f32[1, 192, 1, 1]", le_57: "b8[32, 192, 17, 17]", unsqueeze_1073: "f32[1, 192, 1, 1]", unsqueeze_1085: "f32[1, 128, 1, 1]", unsqueeze_1097: "f32[1, 128, 1, 1]", unsqueeze_1109: "f32[1, 128, 1, 1]", unsqueeze_1121: "f32[1, 128, 1, 1]", le_62: "b8[32, 192, 17, 17]", unsqueeze_1133: "f32[1, 192, 1, 1]", unsqueeze_1145: "f32[1, 128, 1, 1]", unsqueeze_1157: "f32[1, 128, 1, 1]", le_65: "b8[32, 192, 17, 17]", unsqueeze_1169: "f32[1, 192, 1, 1]", le_66: "b8[32, 96, 17, 17]", unsqueeze_1181: "f32[1, 96, 1, 1]", unsqueeze_1193: "f32[1, 96, 1, 1]", unsqueeze_1205: "f32[1, 64, 1, 1]", le_69: "b8[32, 384, 17, 17]", unsqueeze_1217: "f32[1, 384, 1, 1]", le_70: "b8[32, 64, 35, 35]", unsqueeze_1229: "f32[1, 64, 1, 1]", le_71: "b8[32, 96, 35, 35]", unsqueeze_1241: "f32[1, 96, 1, 1]", unsqueeze_1253: "f32[1, 96, 1, 1]", unsqueeze_1265: "f32[1, 64, 1, 1]", le_74: "b8[32, 64, 35, 35]", unsqueeze_1277: "f32[1, 64, 1, 1]", unsqueeze_1289: "f32[1, 48, 1, 1]", le_76: "b8[32, 64, 35, 35]", unsqueeze_1301: "f32[1, 64, 1, 1]", le_77: "b8[32, 64, 35, 35]", unsqueeze_1313: "f32[1, 64, 1, 1]", le_78: "b8[32, 96, 35, 35]", unsqueeze_1325: "f32[1, 96, 1, 1]", unsqueeze_1337: "f32[1, 96, 1, 1]", unsqueeze_1349: "f32[1, 64, 1, 1]", le_81: "b8[32, 64, 35, 35]", unsqueeze_1361: "f32[1, 64, 1, 1]", unsqueeze_1373: "f32[1, 48, 1, 1]", le_83: "b8[32, 64, 35, 35]", unsqueeze_1385: "f32[1, 64, 1, 1]", le_84: "b8[32, 32, 35, 35]", unsqueeze_1397: "f32[1, 32, 1, 1]", le_85: "b8[32, 96, 35, 35]", unsqueeze_1409: "f32[1, 96, 1, 1]", unsqueeze_1421: "f32[1, 96, 1, 1]", unsqueeze_1433: "f32[1, 64, 1, 1]", le_88: "b8[32, 64, 35, 35]", unsqueeze_1445: "f32[1, 64, 1, 1]", unsqueeze_1457: "f32[1, 48, 1, 1]", le_90: "b8[32, 64, 35, 35]", unsqueeze_1469: "f32[1, 64, 1, 1]", unsqueeze_1481: "f32[1, 192, 1, 1]", unsqueeze_1493: "f32[1, 80, 1, 1]", unsqueeze_1505: "f32[1, 64, 1, 1]", unsqueeze_1517: "f32[1, 32, 1, 1]", unsqueeze_1529: "f32[1, 32, 1, 1]", tangents_1: "f32[32, 1000]", tangents_2: "f32[32, 1000]"):
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:153 in _forward, code: x = self.fc(x)
        mm: "f32[32, 2048]" = torch.ops.aten.mm.default(tangents_1, permute_2);  permute_2 = None
        permute_3: "f32[1000, 32]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "f32[1000, 2048]" = torch.ops.aten.mm.default(permute_3, view_1);  permute_3 = view_1 = None
        permute_4: "f32[2048, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_2: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
        permute_5: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:151 in _forward, code: x = torch.flatten(x, 1)
        view_3: "f32[32, 2048, 1, 1]" = torch.ops.aten.view.default(mm, [32, 2048, 1, 1]);  mm = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:149 in _forward, code: x = self.dropout(x)
        convert_element_type: "f32[32, 2048, 1, 1]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_677: "f32[32, 2048, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type, 2.0);  convert_element_type = None
        mul_678: "f32[32, 2048, 1, 1]" = torch.ops.aten.mul.Tensor(view_3, mul_677);  view_3 = mul_677 = None
        clone: "f32[32, 2048, 1, 1]" = torch.ops.aten.clone.default(mul_678, memory_format = torch.contiguous_format);  mul_678 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:147 in _forward, code: x = self.avgpool(x)
        expand: "f32[32, 2048, 8, 8]" = torch.ops.aten.expand.default(clone, [32, 2048, 8, 8]);  clone = None
        div: "f32[32, 2048, 8, 8]" = torch.ops.aten.div.Scalar(expand, 64);  expand = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:364 in forward, code: return torch.cat(outputs, 1)
        slice_4: "f32[32, 320, 8, 8]" = torch.ops.aten.slice.Tensor(div, 1, 0, 320)
        slice_5: "f32[32, 768, 8, 8]" = torch.ops.aten.slice.Tensor(div, 1, 320, 1088)
        slice_6: "f32[32, 768, 8, 8]" = torch.ops.aten.slice.Tensor(div, 1, 1088, 1856)
        slice_7: "f32[32, 192, 8, 8]" = torch.ops.aten.slice.Tensor(div, 1, 1856, 2048);  div = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[32, 192, 8, 8]" = torch.ops.aten.where.self(le, full_default, slice_7);  le = slice_7 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_2: "f32[192]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        sub_96: "f32[32, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_389);  convolution_95 = unsqueeze_389 = None
        mul_679: "f32[32, 192, 8, 8]" = torch.ops.aten.mul.Tensor(where, sub_96)
        sum_3: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_679, [0, 2, 3]);  mul_679 = None
        mul_680: "f32[192]" = torch.ops.aten.mul.Tensor(sum_2, 0.00048828125)
        unsqueeze_390: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_680, 0);  mul_680 = None
        unsqueeze_391: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 2);  unsqueeze_390 = None
        unsqueeze_392: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 3);  unsqueeze_391 = None
        mul_681: "f32[192]" = torch.ops.aten.mul.Tensor(sum_3, 0.00048828125)
        mul_682: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_286, squeeze_286)
        mul_683: "f32[192]" = torch.ops.aten.mul.Tensor(mul_681, mul_682);  mul_681 = mul_682 = None
        unsqueeze_393: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_683, 0);  mul_683 = None
        unsqueeze_394: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 2);  unsqueeze_393 = None
        unsqueeze_395: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 3);  unsqueeze_394 = None
        mul_684: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_286, primals_289);  primals_289 = None
        unsqueeze_396: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
        unsqueeze_397: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 2);  unsqueeze_396 = None
        unsqueeze_398: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 3);  unsqueeze_397 = None
        mul_685: "f32[32, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_395);  sub_96 = unsqueeze_395 = None
        sub_98: "f32[32, 192, 8, 8]" = torch.ops.aten.sub.Tensor(where, mul_685);  where = mul_685 = None
        sub_99: "f32[32, 192, 8, 8]" = torch.ops.aten.sub.Tensor(sub_98, unsqueeze_392);  sub_98 = unsqueeze_392 = None
        mul_686: "f32[32, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_398);  sub_99 = unsqueeze_398 = None
        mul_687: "f32[192]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_286);  sum_3 = squeeze_286 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_686, avg_pool2d_9, primals_288, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_686 = avg_pool2d_9 = primals_288 = None
        getitem_200: "f32[32, 2048, 8, 8]" = convolution_backward[0]
        getitem_201: "f32[192, 2048, 1, 1]" = convolution_backward[1];  convolution_backward = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:356 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward: "f32[32, 2048, 8, 8]" = torch.ops.aten.avg_pool2d_backward.default(getitem_200, cat_12, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_200 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:354 in _forward, code: branch3x3dbl = torch.cat(branch3x3dbl, 1)
        slice_8: "f32[32, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_6, 1, 0, 384)
        slice_9: "f32[32, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_6, 1, 384, 768);  slice_6 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_1: "f32[32, 384, 8, 8]" = torch.ops.aten.where.self(le_1, full_default, slice_9);  le_1 = slice_9 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_4: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        sub_100: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_401);  convolution_94 = unsqueeze_401 = None
        mul_688: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_1, sub_100)
        sum_5: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_688, [0, 2, 3]);  mul_688 = None
        mul_689: "f32[384]" = torch.ops.aten.mul.Tensor(sum_4, 0.00048828125)
        unsqueeze_402: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_689, 0);  mul_689 = None
        unsqueeze_403: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 2);  unsqueeze_402 = None
        unsqueeze_404: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 3);  unsqueeze_403 = None
        mul_690: "f32[384]" = torch.ops.aten.mul.Tensor(sum_5, 0.00048828125)
        mul_691: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_283, squeeze_283)
        mul_692: "f32[384]" = torch.ops.aten.mul.Tensor(mul_690, mul_691);  mul_690 = mul_691 = None
        unsqueeze_405: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_692, 0);  mul_692 = None
        unsqueeze_406: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
        unsqueeze_407: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 3);  unsqueeze_406 = None
        mul_693: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_283, primals_286);  primals_286 = None
        unsqueeze_408: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_693, 0);  mul_693 = None
        unsqueeze_409: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 2);  unsqueeze_408 = None
        unsqueeze_410: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 3);  unsqueeze_409 = None
        mul_694: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_407);  sub_100 = unsqueeze_407 = None
        sub_102: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_1, mul_694);  where_1 = mul_694 = None
        sub_103: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_102, unsqueeze_404);  sub_102 = unsqueeze_404 = None
        mul_695: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_410);  sub_103 = unsqueeze_410 = None
        mul_696: "f32[384]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_283);  sum_5 = squeeze_283 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_695, relu_92, primals_285, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_695 = primals_285 = None
        getitem_203: "f32[32, 384, 8, 8]" = convolution_backward_1[0]
        getitem_204: "f32[384, 384, 3, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_2: "f32[32, 384, 8, 8]" = torch.ops.aten.where.self(le_2, full_default, slice_8);  le_2 = slice_8 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_6: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        sub_104: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_413);  convolution_93 = unsqueeze_413 = None
        mul_697: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_2, sub_104)
        sum_7: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_697, [0, 2, 3]);  mul_697 = None
        mul_698: "f32[384]" = torch.ops.aten.mul.Tensor(sum_6, 0.00048828125)
        unsqueeze_414: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_698, 0);  mul_698 = None
        unsqueeze_415: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 2);  unsqueeze_414 = None
        unsqueeze_416: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 3);  unsqueeze_415 = None
        mul_699: "f32[384]" = torch.ops.aten.mul.Tensor(sum_7, 0.00048828125)
        mul_700: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_280, squeeze_280)
        mul_701: "f32[384]" = torch.ops.aten.mul.Tensor(mul_699, mul_700);  mul_699 = mul_700 = None
        unsqueeze_417: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_701, 0);  mul_701 = None
        unsqueeze_418: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
        unsqueeze_419: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 3);  unsqueeze_418 = None
        mul_702: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_280, primals_283);  primals_283 = None
        unsqueeze_420: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_702, 0);  mul_702 = None
        unsqueeze_421: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 2);  unsqueeze_420 = None
        unsqueeze_422: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 3);  unsqueeze_421 = None
        mul_703: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_419);  sub_104 = unsqueeze_419 = None
        sub_106: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_2, mul_703);  where_2 = mul_703 = None
        sub_107: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_106, unsqueeze_416);  sub_106 = unsqueeze_416 = None
        mul_704: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_422);  sub_107 = unsqueeze_422 = None
        mul_705: "f32[384]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_280);  sum_7 = squeeze_280 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_704, relu_92, primals_282, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_704 = primals_282 = None
        getitem_206: "f32[32, 384, 8, 8]" = convolution_backward_2[0]
        getitem_207: "f32[384, 384, 1, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_483: "f32[32, 384, 8, 8]" = torch.ops.aten.add.Tensor(getitem_203, getitem_206);  getitem_203 = getitem_206 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_206: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(relu_92);  relu_92 = None
        alias_207: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(alias_206);  alias_206 = None
        le_3: "b8[32, 384, 8, 8]" = torch.ops.aten.le.Scalar(alias_207, 0);  alias_207 = None
        where_3: "f32[32, 384, 8, 8]" = torch.ops.aten.where.self(le_3, full_default, add_483);  le_3 = add_483 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_8: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        sub_108: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_425);  convolution_92 = unsqueeze_425 = None
        mul_706: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_3, sub_108)
        sum_9: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_706, [0, 2, 3]);  mul_706 = None
        mul_707: "f32[384]" = torch.ops.aten.mul.Tensor(sum_8, 0.00048828125)
        unsqueeze_426: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_707, 0);  mul_707 = None
        unsqueeze_427: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 2);  unsqueeze_426 = None
        unsqueeze_428: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 3);  unsqueeze_427 = None
        mul_708: "f32[384]" = torch.ops.aten.mul.Tensor(sum_9, 0.00048828125)
        mul_709: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_277, squeeze_277)
        mul_710: "f32[384]" = torch.ops.aten.mul.Tensor(mul_708, mul_709);  mul_708 = mul_709 = None
        unsqueeze_429: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_710, 0);  mul_710 = None
        unsqueeze_430: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
        unsqueeze_431: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
        mul_711: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_277, primals_280);  primals_280 = None
        unsqueeze_432: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
        unsqueeze_433: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 2);  unsqueeze_432 = None
        unsqueeze_434: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 3);  unsqueeze_433 = None
        mul_712: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_431);  sub_108 = unsqueeze_431 = None
        sub_110: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_3, mul_712);  where_3 = mul_712 = None
        sub_111: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_110, unsqueeze_428);  sub_110 = unsqueeze_428 = None
        mul_713: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_434);  sub_111 = unsqueeze_434 = None
        mul_714: "f32[384]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_277);  sum_9 = squeeze_277 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_713, relu_91, primals_279, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_713 = primals_279 = None
        getitem_209: "f32[32, 448, 8, 8]" = convolution_backward_3[0]
        getitem_210: "f32[384, 448, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_210: "f32[32, 448, 8, 8]" = torch.ops.aten.alias.default(relu_91);  relu_91 = None
        alias_211: "f32[32, 448, 8, 8]" = torch.ops.aten.alias.default(alias_210);  alias_210 = None
        le_4: "b8[32, 448, 8, 8]" = torch.ops.aten.le.Scalar(alias_211, 0);  alias_211 = None
        where_4: "f32[32, 448, 8, 8]" = torch.ops.aten.where.self(le_4, full_default, getitem_209);  le_4 = getitem_209 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_10: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        sub_112: "f32[32, 448, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_437);  convolution_91 = unsqueeze_437 = None
        mul_715: "f32[32, 448, 8, 8]" = torch.ops.aten.mul.Tensor(where_4, sub_112)
        sum_11: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_715, [0, 2, 3]);  mul_715 = None
        mul_716: "f32[448]" = torch.ops.aten.mul.Tensor(sum_10, 0.00048828125)
        unsqueeze_438: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_716, 0);  mul_716 = None
        unsqueeze_439: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 2);  unsqueeze_438 = None
        unsqueeze_440: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 3);  unsqueeze_439 = None
        mul_717: "f32[448]" = torch.ops.aten.mul.Tensor(sum_11, 0.00048828125)
        mul_718: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_274, squeeze_274)
        mul_719: "f32[448]" = torch.ops.aten.mul.Tensor(mul_717, mul_718);  mul_717 = mul_718 = None
        unsqueeze_441: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_719, 0);  mul_719 = None
        unsqueeze_442: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
        unsqueeze_443: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 3);  unsqueeze_442 = None
        mul_720: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_274, primals_277);  primals_277 = None
        unsqueeze_444: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
        unsqueeze_445: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 2);  unsqueeze_444 = None
        unsqueeze_446: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 3);  unsqueeze_445 = None
        mul_721: "f32[32, 448, 8, 8]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_443);  sub_112 = unsqueeze_443 = None
        sub_114: "f32[32, 448, 8, 8]" = torch.ops.aten.sub.Tensor(where_4, mul_721);  where_4 = mul_721 = None
        sub_115: "f32[32, 448, 8, 8]" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_440);  sub_114 = unsqueeze_440 = None
        mul_722: "f32[32, 448, 8, 8]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_446);  sub_115 = unsqueeze_446 = None
        mul_723: "f32[448]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_274);  sum_11 = squeeze_274 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_722, cat_12, primals_276, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_722 = primals_276 = None
        getitem_212: "f32[32, 2048, 8, 8]" = convolution_backward_4[0]
        getitem_213: "f32[448, 2048, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_484: "f32[32, 2048, 8, 8]" = torch.ops.aten.add.Tensor(avg_pool2d_backward, getitem_212);  avg_pool2d_backward = getitem_212 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:346 in _forward, code: branch3x3 = torch.cat(branch3x3, 1)
        slice_10: "f32[32, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 384)
        slice_11: "f32[32, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_5, 1, 384, 768);  slice_5 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_5: "f32[32, 384, 8, 8]" = torch.ops.aten.where.self(le_5, full_default, slice_11);  le_5 = slice_11 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_12: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        sub_116: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_449);  convolution_90 = unsqueeze_449 = None
        mul_724: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_5, sub_116)
        sum_13: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_724, [0, 2, 3]);  mul_724 = None
        mul_725: "f32[384]" = torch.ops.aten.mul.Tensor(sum_12, 0.00048828125)
        unsqueeze_450: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_725, 0);  mul_725 = None
        unsqueeze_451: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 2);  unsqueeze_450 = None
        unsqueeze_452: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 3);  unsqueeze_451 = None
        mul_726: "f32[384]" = torch.ops.aten.mul.Tensor(sum_13, 0.00048828125)
        mul_727: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_271, squeeze_271)
        mul_728: "f32[384]" = torch.ops.aten.mul.Tensor(mul_726, mul_727);  mul_726 = mul_727 = None
        unsqueeze_453: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_728, 0);  mul_728 = None
        unsqueeze_454: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 2);  unsqueeze_453 = None
        unsqueeze_455: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 3);  unsqueeze_454 = None
        mul_729: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_271, primals_274);  primals_274 = None
        unsqueeze_456: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_729, 0);  mul_729 = None
        unsqueeze_457: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 2);  unsqueeze_456 = None
        unsqueeze_458: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 3);  unsqueeze_457 = None
        mul_730: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_455);  sub_116 = unsqueeze_455 = None
        sub_118: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_5, mul_730);  where_5 = mul_730 = None
        sub_119: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_118, unsqueeze_452);  sub_118 = unsqueeze_452 = None
        mul_731: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_458);  sub_119 = unsqueeze_458 = None
        mul_732: "f32[384]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_271);  sum_13 = squeeze_271 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_731, relu_88, primals_273, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_731 = primals_273 = None
        getitem_215: "f32[32, 384, 8, 8]" = convolution_backward_5[0]
        getitem_216: "f32[384, 384, 3, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_6: "f32[32, 384, 8, 8]" = torch.ops.aten.where.self(le_6, full_default, slice_10);  le_6 = slice_10 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_14: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        sub_120: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_461);  convolution_89 = unsqueeze_461 = None
        mul_733: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_6, sub_120)
        sum_15: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_733, [0, 2, 3]);  mul_733 = None
        mul_734: "f32[384]" = torch.ops.aten.mul.Tensor(sum_14, 0.00048828125)
        unsqueeze_462: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_734, 0);  mul_734 = None
        unsqueeze_463: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 2);  unsqueeze_462 = None
        unsqueeze_464: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 3);  unsqueeze_463 = None
        mul_735: "f32[384]" = torch.ops.aten.mul.Tensor(sum_15, 0.00048828125)
        mul_736: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_268, squeeze_268)
        mul_737: "f32[384]" = torch.ops.aten.mul.Tensor(mul_735, mul_736);  mul_735 = mul_736 = None
        unsqueeze_465: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_737, 0);  mul_737 = None
        unsqueeze_466: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 2);  unsqueeze_465 = None
        unsqueeze_467: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 3);  unsqueeze_466 = None
        mul_738: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_268, primals_271);  primals_271 = None
        unsqueeze_468: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
        unsqueeze_469: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 2);  unsqueeze_468 = None
        unsqueeze_470: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 3);  unsqueeze_469 = None
        mul_739: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_467);  sub_120 = unsqueeze_467 = None
        sub_122: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_6, mul_739);  where_6 = mul_739 = None
        sub_123: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_464);  sub_122 = unsqueeze_464 = None
        mul_740: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_470);  sub_123 = unsqueeze_470 = None
        mul_741: "f32[384]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_268);  sum_15 = squeeze_268 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_740, relu_88, primals_270, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_740 = primals_270 = None
        getitem_218: "f32[32, 384, 8, 8]" = convolution_backward_6[0]
        getitem_219: "f32[384, 384, 1, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_485: "f32[32, 384, 8, 8]" = torch.ops.aten.add.Tensor(getitem_215, getitem_218);  getitem_215 = getitem_218 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_222: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(relu_88);  relu_88 = None
        alias_223: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(alias_222);  alias_222 = None
        le_7: "b8[32, 384, 8, 8]" = torch.ops.aten.le.Scalar(alias_223, 0);  alias_223 = None
        where_7: "f32[32, 384, 8, 8]" = torch.ops.aten.where.self(le_7, full_default, add_485);  le_7 = add_485 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_16: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
        sub_124: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_473);  convolution_88 = unsqueeze_473 = None
        mul_742: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_7, sub_124)
        sum_17: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_742, [0, 2, 3]);  mul_742 = None
        mul_743: "f32[384]" = torch.ops.aten.mul.Tensor(sum_16, 0.00048828125)
        unsqueeze_474: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
        unsqueeze_475: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 2);  unsqueeze_474 = None
        unsqueeze_476: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 3);  unsqueeze_475 = None
        mul_744: "f32[384]" = torch.ops.aten.mul.Tensor(sum_17, 0.00048828125)
        mul_745: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_265, squeeze_265)
        mul_746: "f32[384]" = torch.ops.aten.mul.Tensor(mul_744, mul_745);  mul_744 = mul_745 = None
        unsqueeze_477: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_746, 0);  mul_746 = None
        unsqueeze_478: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 2);  unsqueeze_477 = None
        unsqueeze_479: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 3);  unsqueeze_478 = None
        mul_747: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_265, primals_268);  primals_268 = None
        unsqueeze_480: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_747, 0);  mul_747 = None
        unsqueeze_481: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 2);  unsqueeze_480 = None
        unsqueeze_482: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 3);  unsqueeze_481 = None
        mul_748: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_479);  sub_124 = unsqueeze_479 = None
        sub_126: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_7, mul_748);  where_7 = mul_748 = None
        sub_127: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_476);  sub_126 = unsqueeze_476 = None
        mul_749: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_482);  sub_127 = unsqueeze_482 = None
        mul_750: "f32[384]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_265);  sum_17 = squeeze_265 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_749, cat_12, primals_267, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_749 = primals_267 = None
        getitem_221: "f32[32, 2048, 8, 8]" = convolution_backward_7[0]
        getitem_222: "f32[384, 2048, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_486: "f32[32, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_484, getitem_221);  add_484 = getitem_221 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_8: "f32[32, 320, 8, 8]" = torch.ops.aten.where.self(le_8, full_default, slice_4);  le_8 = slice_4 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_18: "f32[320]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
        sub_128: "f32[32, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_485);  convolution_87 = unsqueeze_485 = None
        mul_751: "f32[32, 320, 8, 8]" = torch.ops.aten.mul.Tensor(where_8, sub_128)
        sum_19: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_751, [0, 2, 3]);  mul_751 = None
        mul_752: "f32[320]" = torch.ops.aten.mul.Tensor(sum_18, 0.00048828125)
        unsqueeze_486: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_752, 0);  mul_752 = None
        unsqueeze_487: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 2);  unsqueeze_486 = None
        unsqueeze_488: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 3);  unsqueeze_487 = None
        mul_753: "f32[320]" = torch.ops.aten.mul.Tensor(sum_19, 0.00048828125)
        mul_754: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_262, squeeze_262)
        mul_755: "f32[320]" = torch.ops.aten.mul.Tensor(mul_753, mul_754);  mul_753 = mul_754 = None
        unsqueeze_489: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
        unsqueeze_490: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 2);  unsqueeze_489 = None
        unsqueeze_491: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 3);  unsqueeze_490 = None
        mul_756: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_262, primals_265);  primals_265 = None
        unsqueeze_492: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_756, 0);  mul_756 = None
        unsqueeze_493: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 2);  unsqueeze_492 = None
        unsqueeze_494: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 3);  unsqueeze_493 = None
        mul_757: "f32[32, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_491);  sub_128 = unsqueeze_491 = None
        sub_130: "f32[32, 320, 8, 8]" = torch.ops.aten.sub.Tensor(where_8, mul_757);  where_8 = mul_757 = None
        sub_131: "f32[32, 320, 8, 8]" = torch.ops.aten.sub.Tensor(sub_130, unsqueeze_488);  sub_130 = unsqueeze_488 = None
        mul_758: "f32[32, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_494);  sub_131 = unsqueeze_494 = None
        mul_759: "f32[320]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_262);  sum_19 = squeeze_262 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_758, cat_12, primals_264, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_758 = cat_12 = primals_264 = None
        getitem_224: "f32[32, 2048, 8, 8]" = convolution_backward_8[0]
        getitem_225: "f32[320, 2048, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_487: "f32[32, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_486, getitem_224);  add_486 = getitem_224 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:364 in forward, code: return torch.cat(outputs, 1)
        slice_12: "f32[32, 320, 8, 8]" = torch.ops.aten.slice.Tensor(add_487, 1, 0, 320)
        slice_13: "f32[32, 768, 8, 8]" = torch.ops.aten.slice.Tensor(add_487, 1, 320, 1088)
        slice_14: "f32[32, 768, 8, 8]" = torch.ops.aten.slice.Tensor(add_487, 1, 1088, 1856)
        slice_15: "f32[32, 192, 8, 8]" = torch.ops.aten.slice.Tensor(add_487, 1, 1856, 2048);  add_487 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_9: "f32[32, 192, 8, 8]" = torch.ops.aten.where.self(le_9, full_default, slice_15);  le_9 = slice_15 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_20: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
        sub_132: "f32[32, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_497);  convolution_86 = unsqueeze_497 = None
        mul_760: "f32[32, 192, 8, 8]" = torch.ops.aten.mul.Tensor(where_9, sub_132)
        sum_21: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_760, [0, 2, 3]);  mul_760 = None
        mul_761: "f32[192]" = torch.ops.aten.mul.Tensor(sum_20, 0.00048828125)
        unsqueeze_498: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_761, 0);  mul_761 = None
        unsqueeze_499: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 2);  unsqueeze_498 = None
        unsqueeze_500: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 3);  unsqueeze_499 = None
        mul_762: "f32[192]" = torch.ops.aten.mul.Tensor(sum_21, 0.00048828125)
        mul_763: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_259, squeeze_259)
        mul_764: "f32[192]" = torch.ops.aten.mul.Tensor(mul_762, mul_763);  mul_762 = mul_763 = None
        unsqueeze_501: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
        unsqueeze_502: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 2);  unsqueeze_501 = None
        unsqueeze_503: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 3);  unsqueeze_502 = None
        mul_765: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_259, primals_262);  primals_262 = None
        unsqueeze_504: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_765, 0);  mul_765 = None
        unsqueeze_505: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 2);  unsqueeze_504 = None
        unsqueeze_506: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 3);  unsqueeze_505 = None
        mul_766: "f32[32, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_503);  sub_132 = unsqueeze_503 = None
        sub_134: "f32[32, 192, 8, 8]" = torch.ops.aten.sub.Tensor(where_9, mul_766);  where_9 = mul_766 = None
        sub_135: "f32[32, 192, 8, 8]" = torch.ops.aten.sub.Tensor(sub_134, unsqueeze_500);  sub_134 = unsqueeze_500 = None
        mul_767: "f32[32, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_506);  sub_135 = unsqueeze_506 = None
        mul_768: "f32[192]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_259);  sum_21 = squeeze_259 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_767, avg_pool2d_8, primals_261, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_767 = avg_pool2d_8 = primals_261 = None
        getitem_227: "f32[32, 1280, 8, 8]" = convolution_backward_9[0]
        getitem_228: "f32[192, 1280, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:356 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward_1: "f32[32, 1280, 8, 8]" = torch.ops.aten.avg_pool2d_backward.default(getitem_227, cat_9, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_227 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:354 in _forward, code: branch3x3dbl = torch.cat(branch3x3dbl, 1)
        slice_16: "f32[32, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_14, 1, 0, 384)
        slice_17: "f32[32, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_14, 1, 384, 768);  slice_14 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_10: "f32[32, 384, 8, 8]" = torch.ops.aten.where.self(le_10, full_default, slice_17);  le_10 = slice_17 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_22: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        sub_136: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_509);  convolution_85 = unsqueeze_509 = None
        mul_769: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_10, sub_136)
        sum_23: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_769, [0, 2, 3]);  mul_769 = None
        mul_770: "f32[384]" = torch.ops.aten.mul.Tensor(sum_22, 0.00048828125)
        unsqueeze_510: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_770, 0);  mul_770 = None
        unsqueeze_511: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 2);  unsqueeze_510 = None
        unsqueeze_512: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 3);  unsqueeze_511 = None
        mul_771: "f32[384]" = torch.ops.aten.mul.Tensor(sum_23, 0.00048828125)
        mul_772: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_256, squeeze_256)
        mul_773: "f32[384]" = torch.ops.aten.mul.Tensor(mul_771, mul_772);  mul_771 = mul_772 = None
        unsqueeze_513: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
        unsqueeze_514: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 2);  unsqueeze_513 = None
        unsqueeze_515: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 3);  unsqueeze_514 = None
        mul_774: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_256, primals_259);  primals_259 = None
        unsqueeze_516: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
        unsqueeze_517: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 2);  unsqueeze_516 = None
        unsqueeze_518: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 3);  unsqueeze_517 = None
        mul_775: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_515);  sub_136 = unsqueeze_515 = None
        sub_138: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_10, mul_775);  where_10 = mul_775 = None
        sub_139: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_138, unsqueeze_512);  sub_138 = unsqueeze_512 = None
        mul_776: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_518);  sub_139 = unsqueeze_518 = None
        mul_777: "f32[384]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_256);  sum_23 = squeeze_256 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_776, relu_83, primals_258, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_776 = primals_258 = None
        getitem_230: "f32[32, 384, 8, 8]" = convolution_backward_10[0]
        getitem_231: "f32[384, 384, 3, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_11: "f32[32, 384, 8, 8]" = torch.ops.aten.where.self(le_11, full_default, slice_16);  le_11 = slice_16 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_24: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        sub_140: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_521);  convolution_84 = unsqueeze_521 = None
        mul_778: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_11, sub_140)
        sum_25: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_778, [0, 2, 3]);  mul_778 = None
        mul_779: "f32[384]" = torch.ops.aten.mul.Tensor(sum_24, 0.00048828125)
        unsqueeze_522: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_779, 0);  mul_779 = None
        unsqueeze_523: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 2);  unsqueeze_522 = None
        unsqueeze_524: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 3);  unsqueeze_523 = None
        mul_780: "f32[384]" = torch.ops.aten.mul.Tensor(sum_25, 0.00048828125)
        mul_781: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_253, squeeze_253)
        mul_782: "f32[384]" = torch.ops.aten.mul.Tensor(mul_780, mul_781);  mul_780 = mul_781 = None
        unsqueeze_525: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_782, 0);  mul_782 = None
        unsqueeze_526: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 2);  unsqueeze_525 = None
        unsqueeze_527: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 3);  unsqueeze_526 = None
        mul_783: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_253, primals_256);  primals_256 = None
        unsqueeze_528: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
        unsqueeze_529: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 2);  unsqueeze_528 = None
        unsqueeze_530: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 3);  unsqueeze_529 = None
        mul_784: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_527);  sub_140 = unsqueeze_527 = None
        sub_142: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_11, mul_784);  where_11 = mul_784 = None
        sub_143: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_524);  sub_142 = unsqueeze_524 = None
        mul_785: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_530);  sub_143 = unsqueeze_530 = None
        mul_786: "f32[384]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_253);  sum_25 = squeeze_253 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_785, relu_83, primals_255, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_785 = primals_255 = None
        getitem_233: "f32[32, 384, 8, 8]" = convolution_backward_11[0]
        getitem_234: "f32[384, 384, 1, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_488: "f32[32, 384, 8, 8]" = torch.ops.aten.add.Tensor(getitem_230, getitem_233);  getitem_230 = getitem_233 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_242: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(relu_83);  relu_83 = None
        alias_243: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(alias_242);  alias_242 = None
        le_12: "b8[32, 384, 8, 8]" = torch.ops.aten.le.Scalar(alias_243, 0);  alias_243 = None
        where_12: "f32[32, 384, 8, 8]" = torch.ops.aten.where.self(le_12, full_default, add_488);  le_12 = add_488 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_26: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
        sub_144: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_533);  convolution_83 = unsqueeze_533 = None
        mul_787: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_12, sub_144)
        sum_27: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_787, [0, 2, 3]);  mul_787 = None
        mul_788: "f32[384]" = torch.ops.aten.mul.Tensor(sum_26, 0.00048828125)
        unsqueeze_534: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_788, 0);  mul_788 = None
        unsqueeze_535: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 2);  unsqueeze_534 = None
        unsqueeze_536: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 3);  unsqueeze_535 = None
        mul_789: "f32[384]" = torch.ops.aten.mul.Tensor(sum_27, 0.00048828125)
        mul_790: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_250, squeeze_250)
        mul_791: "f32[384]" = torch.ops.aten.mul.Tensor(mul_789, mul_790);  mul_789 = mul_790 = None
        unsqueeze_537: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_791, 0);  mul_791 = None
        unsqueeze_538: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 2);  unsqueeze_537 = None
        unsqueeze_539: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 3);  unsqueeze_538 = None
        mul_792: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_250, primals_253);  primals_253 = None
        unsqueeze_540: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_792, 0);  mul_792 = None
        unsqueeze_541: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 2);  unsqueeze_540 = None
        unsqueeze_542: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 3);  unsqueeze_541 = None
        mul_793: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_539);  sub_144 = unsqueeze_539 = None
        sub_146: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_12, mul_793);  where_12 = mul_793 = None
        sub_147: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_146, unsqueeze_536);  sub_146 = unsqueeze_536 = None
        mul_794: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_542);  sub_147 = unsqueeze_542 = None
        mul_795: "f32[384]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_250);  sum_27 = squeeze_250 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_794, relu_82, primals_252, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_794 = primals_252 = None
        getitem_236: "f32[32, 448, 8, 8]" = convolution_backward_12[0]
        getitem_237: "f32[384, 448, 3, 3]" = convolution_backward_12[1];  convolution_backward_12 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_246: "f32[32, 448, 8, 8]" = torch.ops.aten.alias.default(relu_82);  relu_82 = None
        alias_247: "f32[32, 448, 8, 8]" = torch.ops.aten.alias.default(alias_246);  alias_246 = None
        le_13: "b8[32, 448, 8, 8]" = torch.ops.aten.le.Scalar(alias_247, 0);  alias_247 = None
        where_13: "f32[32, 448, 8, 8]" = torch.ops.aten.where.self(le_13, full_default, getitem_236);  le_13 = getitem_236 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_28: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
        sub_148: "f32[32, 448, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_545);  convolution_82 = unsqueeze_545 = None
        mul_796: "f32[32, 448, 8, 8]" = torch.ops.aten.mul.Tensor(where_13, sub_148)
        sum_29: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_796, [0, 2, 3]);  mul_796 = None
        mul_797: "f32[448]" = torch.ops.aten.mul.Tensor(sum_28, 0.00048828125)
        unsqueeze_546: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_797, 0);  mul_797 = None
        unsqueeze_547: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 2);  unsqueeze_546 = None
        unsqueeze_548: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 3);  unsqueeze_547 = None
        mul_798: "f32[448]" = torch.ops.aten.mul.Tensor(sum_29, 0.00048828125)
        mul_799: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_247, squeeze_247)
        mul_800: "f32[448]" = torch.ops.aten.mul.Tensor(mul_798, mul_799);  mul_798 = mul_799 = None
        unsqueeze_549: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
        unsqueeze_550: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 2);  unsqueeze_549 = None
        unsqueeze_551: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 3);  unsqueeze_550 = None
        mul_801: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_247, primals_250);  primals_250 = None
        unsqueeze_552: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_801, 0);  mul_801 = None
        unsqueeze_553: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 2);  unsqueeze_552 = None
        unsqueeze_554: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 3);  unsqueeze_553 = None
        mul_802: "f32[32, 448, 8, 8]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_551);  sub_148 = unsqueeze_551 = None
        sub_150: "f32[32, 448, 8, 8]" = torch.ops.aten.sub.Tensor(where_13, mul_802);  where_13 = mul_802 = None
        sub_151: "f32[32, 448, 8, 8]" = torch.ops.aten.sub.Tensor(sub_150, unsqueeze_548);  sub_150 = unsqueeze_548 = None
        mul_803: "f32[32, 448, 8, 8]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_554);  sub_151 = unsqueeze_554 = None
        mul_804: "f32[448]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_247);  sum_29 = squeeze_247 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_803, cat_9, primals_249, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_803 = primals_249 = None
        getitem_239: "f32[32, 1280, 8, 8]" = convolution_backward_13[0]
        getitem_240: "f32[448, 1280, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_489: "f32[32, 1280, 8, 8]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_1, getitem_239);  avg_pool2d_backward_1 = getitem_239 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:346 in _forward, code: branch3x3 = torch.cat(branch3x3, 1)
        slice_18: "f32[32, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 384)
        slice_19: "f32[32, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_13, 1, 384, 768);  slice_13 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_14: "f32[32, 384, 8, 8]" = torch.ops.aten.where.self(le_14, full_default, slice_19);  le_14 = slice_19 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_30: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
        sub_152: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_557);  convolution_81 = unsqueeze_557 = None
        mul_805: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_14, sub_152)
        sum_31: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_805, [0, 2, 3]);  mul_805 = None
        mul_806: "f32[384]" = torch.ops.aten.mul.Tensor(sum_30, 0.00048828125)
        unsqueeze_558: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_806, 0);  mul_806 = None
        unsqueeze_559: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 2);  unsqueeze_558 = None
        unsqueeze_560: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 3);  unsqueeze_559 = None
        mul_807: "f32[384]" = torch.ops.aten.mul.Tensor(sum_31, 0.00048828125)
        mul_808: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_244, squeeze_244)
        mul_809: "f32[384]" = torch.ops.aten.mul.Tensor(mul_807, mul_808);  mul_807 = mul_808 = None
        unsqueeze_561: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_809, 0);  mul_809 = None
        unsqueeze_562: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 2);  unsqueeze_561 = None
        unsqueeze_563: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 3);  unsqueeze_562 = None
        mul_810: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_244, primals_247);  primals_247 = None
        unsqueeze_564: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_810, 0);  mul_810 = None
        unsqueeze_565: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 2);  unsqueeze_564 = None
        unsqueeze_566: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 3);  unsqueeze_565 = None
        mul_811: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_563);  sub_152 = unsqueeze_563 = None
        sub_154: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_14, mul_811);  where_14 = mul_811 = None
        sub_155: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_154, unsqueeze_560);  sub_154 = unsqueeze_560 = None
        mul_812: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_566);  sub_155 = unsqueeze_566 = None
        mul_813: "f32[384]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_244);  sum_31 = squeeze_244 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_812, relu_79, primals_246, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_812 = primals_246 = None
        getitem_242: "f32[32, 384, 8, 8]" = convolution_backward_14[0]
        getitem_243: "f32[384, 384, 3, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_15: "f32[32, 384, 8, 8]" = torch.ops.aten.where.self(le_15, full_default, slice_18);  le_15 = slice_18 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_32: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
        sub_156: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_569);  convolution_80 = unsqueeze_569 = None
        mul_814: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_15, sub_156)
        sum_33: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_814, [0, 2, 3]);  mul_814 = None
        mul_815: "f32[384]" = torch.ops.aten.mul.Tensor(sum_32, 0.00048828125)
        unsqueeze_570: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_815, 0);  mul_815 = None
        unsqueeze_571: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 2);  unsqueeze_570 = None
        unsqueeze_572: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 3);  unsqueeze_571 = None
        mul_816: "f32[384]" = torch.ops.aten.mul.Tensor(sum_33, 0.00048828125)
        mul_817: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_241, squeeze_241)
        mul_818: "f32[384]" = torch.ops.aten.mul.Tensor(mul_816, mul_817);  mul_816 = mul_817 = None
        unsqueeze_573: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_818, 0);  mul_818 = None
        unsqueeze_574: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 2);  unsqueeze_573 = None
        unsqueeze_575: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 3);  unsqueeze_574 = None
        mul_819: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_241, primals_244);  primals_244 = None
        unsqueeze_576: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_819, 0);  mul_819 = None
        unsqueeze_577: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 2);  unsqueeze_576 = None
        unsqueeze_578: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 3);  unsqueeze_577 = None
        mul_820: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_575);  sub_156 = unsqueeze_575 = None
        sub_158: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_15, mul_820);  where_15 = mul_820 = None
        sub_159: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_158, unsqueeze_572);  sub_158 = unsqueeze_572 = None
        mul_821: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_578);  sub_159 = unsqueeze_578 = None
        mul_822: "f32[384]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_241);  sum_33 = squeeze_241 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_821, relu_79, primals_243, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_821 = primals_243 = None
        getitem_245: "f32[32, 384, 8, 8]" = convolution_backward_15[0]
        getitem_246: "f32[384, 384, 1, 3]" = convolution_backward_15[1];  convolution_backward_15 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_490: "f32[32, 384, 8, 8]" = torch.ops.aten.add.Tensor(getitem_242, getitem_245);  getitem_242 = getitem_245 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_258: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(relu_79);  relu_79 = None
        alias_259: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(alias_258);  alias_258 = None
        le_16: "b8[32, 384, 8, 8]" = torch.ops.aten.le.Scalar(alias_259, 0);  alias_259 = None
        where_16: "f32[32, 384, 8, 8]" = torch.ops.aten.where.self(le_16, full_default, add_490);  le_16 = add_490 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_34: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
        sub_160: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_581);  convolution_79 = unsqueeze_581 = None
        mul_823: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_16, sub_160)
        sum_35: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_823, [0, 2, 3]);  mul_823 = None
        mul_824: "f32[384]" = torch.ops.aten.mul.Tensor(sum_34, 0.00048828125)
        unsqueeze_582: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_824, 0);  mul_824 = None
        unsqueeze_583: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 2);  unsqueeze_582 = None
        unsqueeze_584: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 3);  unsqueeze_583 = None
        mul_825: "f32[384]" = torch.ops.aten.mul.Tensor(sum_35, 0.00048828125)
        mul_826: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_238, squeeze_238)
        mul_827: "f32[384]" = torch.ops.aten.mul.Tensor(mul_825, mul_826);  mul_825 = mul_826 = None
        unsqueeze_585: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_827, 0);  mul_827 = None
        unsqueeze_586: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 2);  unsqueeze_585 = None
        unsqueeze_587: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 3);  unsqueeze_586 = None
        mul_828: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_238, primals_241);  primals_241 = None
        unsqueeze_588: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_828, 0);  mul_828 = None
        unsqueeze_589: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 2);  unsqueeze_588 = None
        unsqueeze_590: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 3);  unsqueeze_589 = None
        mul_829: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_587);  sub_160 = unsqueeze_587 = None
        sub_162: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_16, mul_829);  where_16 = mul_829 = None
        sub_163: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_162, unsqueeze_584);  sub_162 = unsqueeze_584 = None
        mul_830: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_590);  sub_163 = unsqueeze_590 = None
        mul_831: "f32[384]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_238);  sum_35 = squeeze_238 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_830, cat_9, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_830 = primals_240 = None
        getitem_248: "f32[32, 1280, 8, 8]" = convolution_backward_16[0]
        getitem_249: "f32[384, 1280, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_491: "f32[32, 1280, 8, 8]" = torch.ops.aten.add.Tensor(add_489, getitem_248);  add_489 = getitem_248 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_17: "f32[32, 320, 8, 8]" = torch.ops.aten.where.self(le_17, full_default, slice_12);  le_17 = slice_12 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_36: "f32[320]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
        sub_164: "f32[32, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_593);  convolution_78 = unsqueeze_593 = None
        mul_832: "f32[32, 320, 8, 8]" = torch.ops.aten.mul.Tensor(where_17, sub_164)
        sum_37: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_832, [0, 2, 3]);  mul_832 = None
        mul_833: "f32[320]" = torch.ops.aten.mul.Tensor(sum_36, 0.00048828125)
        unsqueeze_594: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_833, 0);  mul_833 = None
        unsqueeze_595: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 2);  unsqueeze_594 = None
        unsqueeze_596: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 3);  unsqueeze_595 = None
        mul_834: "f32[320]" = torch.ops.aten.mul.Tensor(sum_37, 0.00048828125)
        mul_835: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_235, squeeze_235)
        mul_836: "f32[320]" = torch.ops.aten.mul.Tensor(mul_834, mul_835);  mul_834 = mul_835 = None
        unsqueeze_597: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_836, 0);  mul_836 = None
        unsqueeze_598: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 2);  unsqueeze_597 = None
        unsqueeze_599: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 3);  unsqueeze_598 = None
        mul_837: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_235, primals_238);  primals_238 = None
        unsqueeze_600: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_837, 0);  mul_837 = None
        unsqueeze_601: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 2);  unsqueeze_600 = None
        unsqueeze_602: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 3);  unsqueeze_601 = None
        mul_838: "f32[32, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_599);  sub_164 = unsqueeze_599 = None
        sub_166: "f32[32, 320, 8, 8]" = torch.ops.aten.sub.Tensor(where_17, mul_838);  where_17 = mul_838 = None
        sub_167: "f32[32, 320, 8, 8]" = torch.ops.aten.sub.Tensor(sub_166, unsqueeze_596);  sub_166 = unsqueeze_596 = None
        mul_839: "f32[32, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_602);  sub_167 = unsqueeze_602 = None
        mul_840: "f32[320]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_235);  sum_37 = squeeze_235 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_839, cat_9, primals_237, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_839 = cat_9 = primals_237 = None
        getitem_251: "f32[32, 1280, 8, 8]" = convolution_backward_17[0]
        getitem_252: "f32[320, 1280, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_492: "f32[32, 1280, 8, 8]" = torch.ops.aten.add.Tensor(add_491, getitem_251);  add_491 = getitem_251 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:317 in forward, code: return torch.cat(outputs, 1)
        slice_20: "f32[32, 320, 8, 8]" = torch.ops.aten.slice.Tensor(add_492, 1, 0, 320)
        slice_21: "f32[32, 192, 8, 8]" = torch.ops.aten.slice.Tensor(add_492, 1, 320, 512)
        slice_22: "f32[32, 768, 8, 8]" = torch.ops.aten.slice.Tensor(add_492, 1, 512, 1280);  add_492 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:311 in _forward, code: branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        max_pool2d_with_indices_backward: "f32[32, 768, 17, 17]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_22, cat_8, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_163);  slice_22 = getitem_163 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_18: "f32[32, 192, 8, 8]" = torch.ops.aten.where.self(le_18, full_default, slice_21);  le_18 = slice_21 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_38: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
        sub_168: "f32[32, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_605);  convolution_77 = unsqueeze_605 = None
        mul_841: "f32[32, 192, 8, 8]" = torch.ops.aten.mul.Tensor(where_18, sub_168)
        sum_39: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_841, [0, 2, 3]);  mul_841 = None
        mul_842: "f32[192]" = torch.ops.aten.mul.Tensor(sum_38, 0.00048828125)
        unsqueeze_606: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_842, 0);  mul_842 = None
        unsqueeze_607: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 2);  unsqueeze_606 = None
        unsqueeze_608: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 3);  unsqueeze_607 = None
        mul_843: "f32[192]" = torch.ops.aten.mul.Tensor(sum_39, 0.00048828125)
        mul_844: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
        mul_845: "f32[192]" = torch.ops.aten.mul.Tensor(mul_843, mul_844);  mul_843 = mul_844 = None
        unsqueeze_609: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_845, 0);  mul_845 = None
        unsqueeze_610: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 2);  unsqueeze_609 = None
        unsqueeze_611: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 3);  unsqueeze_610 = None
        mul_846: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_232, primals_235);  primals_235 = None
        unsqueeze_612: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_846, 0);  mul_846 = None
        unsqueeze_613: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 2);  unsqueeze_612 = None
        unsqueeze_614: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 3);  unsqueeze_613 = None
        mul_847: "f32[32, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_611);  sub_168 = unsqueeze_611 = None
        sub_170: "f32[32, 192, 8, 8]" = torch.ops.aten.sub.Tensor(where_18, mul_847);  where_18 = mul_847 = None
        sub_171: "f32[32, 192, 8, 8]" = torch.ops.aten.sub.Tensor(sub_170, unsqueeze_608);  sub_170 = unsqueeze_608 = None
        mul_848: "f32[32, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_614);  sub_171 = unsqueeze_614 = None
        mul_849: "f32[192]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_232);  sum_39 = squeeze_232 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_848, relu_76, primals_234, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_848 = primals_234 = None
        getitem_254: "f32[32, 192, 17, 17]" = convolution_backward_18[0]
        getitem_255: "f32[192, 192, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_270: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_76);  relu_76 = None
        alias_271: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_270);  alias_270 = None
        le_19: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_271, 0);  alias_271 = None
        where_19: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_19, full_default, getitem_254);  le_19 = getitem_254 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_40: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
        sub_172: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_617);  convolution_76 = unsqueeze_617 = None
        mul_850: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_19, sub_172)
        sum_41: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_850, [0, 2, 3]);  mul_850 = None
        mul_851: "f32[192]" = torch.ops.aten.mul.Tensor(sum_40, 0.00010813148788927336)
        unsqueeze_618: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_851, 0);  mul_851 = None
        unsqueeze_619: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 2);  unsqueeze_618 = None
        unsqueeze_620: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 3);  unsqueeze_619 = None
        mul_852: "f32[192]" = torch.ops.aten.mul.Tensor(sum_41, 0.00010813148788927336)
        mul_853: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_229, squeeze_229)
        mul_854: "f32[192]" = torch.ops.aten.mul.Tensor(mul_852, mul_853);  mul_852 = mul_853 = None
        unsqueeze_621: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_854, 0);  mul_854 = None
        unsqueeze_622: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 2);  unsqueeze_621 = None
        unsqueeze_623: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 3);  unsqueeze_622 = None
        mul_855: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_229, primals_232);  primals_232 = None
        unsqueeze_624: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_855, 0);  mul_855 = None
        unsqueeze_625: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 2);  unsqueeze_624 = None
        unsqueeze_626: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 3);  unsqueeze_625 = None
        mul_856: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_623);  sub_172 = unsqueeze_623 = None
        sub_174: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_19, mul_856);  where_19 = mul_856 = None
        sub_175: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_620);  sub_174 = unsqueeze_620 = None
        mul_857: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_626);  sub_175 = unsqueeze_626 = None
        mul_858: "f32[192]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_229);  sum_41 = squeeze_229 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_857, relu_75, primals_231, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_857 = primals_231 = None
        getitem_257: "f32[32, 192, 17, 17]" = convolution_backward_19[0]
        getitem_258: "f32[192, 192, 7, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_274: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_75);  relu_75 = None
        alias_275: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_274);  alias_274 = None
        le_20: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_275, 0);  alias_275 = None
        where_20: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_20, full_default, getitem_257);  le_20 = getitem_257 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_42: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
        sub_176: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_629);  convolution_75 = unsqueeze_629 = None
        mul_859: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_20, sub_176)
        sum_43: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_859, [0, 2, 3]);  mul_859 = None
        mul_860: "f32[192]" = torch.ops.aten.mul.Tensor(sum_42, 0.00010813148788927336)
        unsqueeze_630: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_860, 0);  mul_860 = None
        unsqueeze_631: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 2);  unsqueeze_630 = None
        unsqueeze_632: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 3);  unsqueeze_631 = None
        mul_861: "f32[192]" = torch.ops.aten.mul.Tensor(sum_43, 0.00010813148788927336)
        mul_862: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_226, squeeze_226)
        mul_863: "f32[192]" = torch.ops.aten.mul.Tensor(mul_861, mul_862);  mul_861 = mul_862 = None
        unsqueeze_633: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_863, 0);  mul_863 = None
        unsqueeze_634: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 2);  unsqueeze_633 = None
        unsqueeze_635: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 3);  unsqueeze_634 = None
        mul_864: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_226, primals_229);  primals_229 = None
        unsqueeze_636: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_864, 0);  mul_864 = None
        unsqueeze_637: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 2);  unsqueeze_636 = None
        unsqueeze_638: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 3);  unsqueeze_637 = None
        mul_865: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_635);  sub_176 = unsqueeze_635 = None
        sub_178: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_20, mul_865);  where_20 = mul_865 = None
        sub_179: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_178, unsqueeze_632);  sub_178 = unsqueeze_632 = None
        mul_866: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_638);  sub_179 = unsqueeze_638 = None
        mul_867: "f32[192]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_226);  sum_43 = squeeze_226 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_866, relu_74, primals_228, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_866 = primals_228 = None
        getitem_260: "f32[32, 192, 17, 17]" = convolution_backward_20[0]
        getitem_261: "f32[192, 192, 1, 7]" = convolution_backward_20[1];  convolution_backward_20 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_278: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_74);  relu_74 = None
        alias_279: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_278);  alias_278 = None
        le_21: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_279, 0);  alias_279 = None
        where_21: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_21, full_default, getitem_260);  le_21 = getitem_260 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_44: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
        sub_180: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_641);  convolution_74 = unsqueeze_641 = None
        mul_868: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_21, sub_180)
        sum_45: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_868, [0, 2, 3]);  mul_868 = None
        mul_869: "f32[192]" = torch.ops.aten.mul.Tensor(sum_44, 0.00010813148788927336)
        unsqueeze_642: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_869, 0);  mul_869 = None
        unsqueeze_643: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 2);  unsqueeze_642 = None
        unsqueeze_644: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 3);  unsqueeze_643 = None
        mul_870: "f32[192]" = torch.ops.aten.mul.Tensor(sum_45, 0.00010813148788927336)
        mul_871: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_223, squeeze_223)
        mul_872: "f32[192]" = torch.ops.aten.mul.Tensor(mul_870, mul_871);  mul_870 = mul_871 = None
        unsqueeze_645: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_872, 0);  mul_872 = None
        unsqueeze_646: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 2);  unsqueeze_645 = None
        unsqueeze_647: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 3);  unsqueeze_646 = None
        mul_873: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_223, primals_226);  primals_226 = None
        unsqueeze_648: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_873, 0);  mul_873 = None
        unsqueeze_649: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 2);  unsqueeze_648 = None
        unsqueeze_650: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 3);  unsqueeze_649 = None
        mul_874: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_647);  sub_180 = unsqueeze_647 = None
        sub_182: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_21, mul_874);  where_21 = mul_874 = None
        sub_183: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_182, unsqueeze_644);  sub_182 = unsqueeze_644 = None
        mul_875: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_650);  sub_183 = unsqueeze_650 = None
        mul_876: "f32[192]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_223);  sum_45 = squeeze_223 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_875, cat_8, primals_225, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_875 = primals_225 = None
        getitem_263: "f32[32, 768, 17, 17]" = convolution_backward_21[0]
        getitem_264: "f32[192, 768, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_493: "f32[32, 768, 17, 17]" = torch.ops.aten.add.Tensor(max_pool2d_with_indices_backward, getitem_263);  max_pool2d_with_indices_backward = getitem_263 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_22: "f32[32, 320, 8, 8]" = torch.ops.aten.where.self(le_22, full_default, slice_20);  le_22 = slice_20 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_46: "f32[320]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
        sub_184: "f32[32, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_653);  convolution_73 = unsqueeze_653 = None
        mul_877: "f32[32, 320, 8, 8]" = torch.ops.aten.mul.Tensor(where_22, sub_184)
        sum_47: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_877, [0, 2, 3]);  mul_877 = None
        mul_878: "f32[320]" = torch.ops.aten.mul.Tensor(sum_46, 0.00048828125)
        unsqueeze_654: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_878, 0);  mul_878 = None
        unsqueeze_655: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 2);  unsqueeze_654 = None
        unsqueeze_656: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 3);  unsqueeze_655 = None
        mul_879: "f32[320]" = torch.ops.aten.mul.Tensor(sum_47, 0.00048828125)
        mul_880: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_220, squeeze_220)
        mul_881: "f32[320]" = torch.ops.aten.mul.Tensor(mul_879, mul_880);  mul_879 = mul_880 = None
        unsqueeze_657: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_881, 0);  mul_881 = None
        unsqueeze_658: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 2);  unsqueeze_657 = None
        unsqueeze_659: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 3);  unsqueeze_658 = None
        mul_882: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_220, primals_223);  primals_223 = None
        unsqueeze_660: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_882, 0);  mul_882 = None
        unsqueeze_661: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, 2);  unsqueeze_660 = None
        unsqueeze_662: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 3);  unsqueeze_661 = None
        mul_883: "f32[32, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_659);  sub_184 = unsqueeze_659 = None
        sub_186: "f32[32, 320, 8, 8]" = torch.ops.aten.sub.Tensor(where_22, mul_883);  where_22 = mul_883 = None
        sub_187: "f32[32, 320, 8, 8]" = torch.ops.aten.sub.Tensor(sub_186, unsqueeze_656);  sub_186 = unsqueeze_656 = None
        mul_884: "f32[32, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_662);  sub_187 = unsqueeze_662 = None
        mul_885: "f32[320]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_220);  sum_47 = squeeze_220 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_884, relu_72, primals_222, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_884 = primals_222 = None
        getitem_266: "f32[32, 192, 17, 17]" = convolution_backward_22[0]
        getitem_267: "f32[320, 192, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_286: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_72);  relu_72 = None
        alias_287: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_286);  alias_286 = None
        le_23: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_287, 0);  alias_287 = None
        where_23: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_23, full_default, getitem_266);  le_23 = getitem_266 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_48: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
        sub_188: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_665);  convolution_72 = unsqueeze_665 = None
        mul_886: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_23, sub_188)
        sum_49: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_886, [0, 2, 3]);  mul_886 = None
        mul_887: "f32[192]" = torch.ops.aten.mul.Tensor(sum_48, 0.00010813148788927336)
        unsqueeze_666: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_887, 0);  mul_887 = None
        unsqueeze_667: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 2);  unsqueeze_666 = None
        unsqueeze_668: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 3);  unsqueeze_667 = None
        mul_888: "f32[192]" = torch.ops.aten.mul.Tensor(sum_49, 0.00010813148788927336)
        mul_889: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_217, squeeze_217)
        mul_890: "f32[192]" = torch.ops.aten.mul.Tensor(mul_888, mul_889);  mul_888 = mul_889 = None
        unsqueeze_669: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_890, 0);  mul_890 = None
        unsqueeze_670: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 2);  unsqueeze_669 = None
        unsqueeze_671: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 3);  unsqueeze_670 = None
        mul_891: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_217, primals_220);  primals_220 = None
        unsqueeze_672: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_891, 0);  mul_891 = None
        unsqueeze_673: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, 2);  unsqueeze_672 = None
        unsqueeze_674: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 3);  unsqueeze_673 = None
        mul_892: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_671);  sub_188 = unsqueeze_671 = None
        sub_190: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_23, mul_892);  where_23 = mul_892 = None
        sub_191: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_190, unsqueeze_668);  sub_190 = unsqueeze_668 = None
        mul_893: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_674);  sub_191 = unsqueeze_674 = None
        mul_894: "f32[192]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_217);  sum_49 = squeeze_217 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_893, cat_8, primals_219, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_893 = primals_219 = None
        getitem_269: "f32[32, 768, 17, 17]" = convolution_backward_23[0]
        getitem_270: "f32[192, 768, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_494: "f32[32, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_493, getitem_269);  add_493 = getitem_269 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:393 in forward, code: x = self.fc(x)
        mm_2: "f32[32, 768]" = torch.ops.aten.mm.default(tangents_2, permute_6);  permute_6 = None
        permute_7: "f32[1000, 32]" = torch.ops.aten.permute.default(tangents_2, [1, 0])
        mm_3: "f32[1000, 768]" = torch.ops.aten.mm.default(permute_7, view);  permute_7 = view = None
        permute_8: "f32[768, 1000]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
        sum_50: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_2, [0], True);  tangents_2 = None
        view_4: "f32[1000]" = torch.ops.aten.view.default(sum_50, [1000]);  sum_50 = None
        permute_9: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:391 in forward, code: x = torch.flatten(x, 1)
        view_5: "f32[32, 768, 1, 1]" = torch.ops.aten.view.default(mm_2, [32, 768, 1, 1]);  mm_2 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:389 in forward, code: x = F.adaptive_avg_pool2d(x, (1, 1))
        expand_1: "f32[32, 768, 1, 1]" = torch.ops.aten.expand.default(view_5, [32, 768, 1, 1]);  view_5 = None
        div_1: "f32[32, 768, 1, 1]" = torch.ops.aten.div.Scalar(expand_1, 1);  expand_1 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_24: "f32[32, 768, 1, 1]" = torch.ops.aten.where.self(le_24, full_default, div_1);  le_24 = div_1 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
        sub_192: "f32[32, 768, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_677);  convolution_71 = unsqueeze_677 = None
        mul_895: "f32[32, 768, 1, 1]" = torch.ops.aten.mul.Tensor(where_24, sub_192)
        sum_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_895, [0, 2, 3]);  mul_895 = None
        mul_896: "f32[768]" = torch.ops.aten.mul.Tensor(sum_51, 0.03125)
        unsqueeze_678: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_896, 0);  mul_896 = None
        unsqueeze_679: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 2);  unsqueeze_678 = None
        unsqueeze_680: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 3);  unsqueeze_679 = None
        mul_897: "f32[768]" = torch.ops.aten.mul.Tensor(sum_52, 0.03125)
        mul_898: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
        mul_899: "f32[768]" = torch.ops.aten.mul.Tensor(mul_897, mul_898);  mul_897 = mul_898 = None
        unsqueeze_681: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_899, 0);  mul_899 = None
        unsqueeze_682: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 2);  unsqueeze_681 = None
        unsqueeze_683: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 3);  unsqueeze_682 = None
        mul_900: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_214, primals_215);  primals_215 = None
        unsqueeze_684: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_900, 0);  mul_900 = None
        unsqueeze_685: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 2);  unsqueeze_684 = None
        unsqueeze_686: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 3);  unsqueeze_685 = None
        mul_901: "f32[32, 768, 1, 1]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_683);  sub_192 = unsqueeze_683 = None
        sub_194: "f32[32, 768, 1, 1]" = torch.ops.aten.sub.Tensor(where_24, mul_901);  where_24 = mul_901 = None
        sub_195: "f32[32, 768, 1, 1]" = torch.ops.aten.sub.Tensor(sub_194, unsqueeze_680);  sub_194 = unsqueeze_680 = None
        mul_902: "f32[32, 768, 1, 1]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_686);  sub_195 = unsqueeze_686 = None
        mul_903: "f32[768]" = torch.ops.aten.mul.Tensor(sum_52, squeeze_214);  sum_52 = squeeze_214 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_902, relu_70, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_902 = primals_214 = None
        getitem_272: "f32[32, 128, 5, 5]" = convolution_backward_24[0]
        getitem_273: "f32[768, 128, 5, 5]" = convolution_backward_24[1];  convolution_backward_24 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_294: "f32[32, 128, 5, 5]" = torch.ops.aten.alias.default(relu_70);  relu_70 = None
        alias_295: "f32[32, 128, 5, 5]" = torch.ops.aten.alias.default(alias_294);  alias_294 = None
        le_25: "b8[32, 128, 5, 5]" = torch.ops.aten.le.Scalar(alias_295, 0);  alias_295 = None
        where_25: "f32[32, 128, 5, 5]" = torch.ops.aten.where.self(le_25, full_default, getitem_272);  le_25 = getitem_272 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_53: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
        sub_196: "f32[32, 128, 5, 5]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_689);  convolution_70 = unsqueeze_689 = None
        mul_904: "f32[32, 128, 5, 5]" = torch.ops.aten.mul.Tensor(where_25, sub_196)
        sum_54: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_904, [0, 2, 3]);  mul_904 = None
        mul_905: "f32[128]" = torch.ops.aten.mul.Tensor(sum_53, 0.00125)
        unsqueeze_690: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_905, 0);  mul_905 = None
        unsqueeze_691: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 2);  unsqueeze_690 = None
        unsqueeze_692: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 3);  unsqueeze_691 = None
        mul_906: "f32[128]" = torch.ops.aten.mul.Tensor(sum_54, 0.00125)
        mul_907: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_211, squeeze_211)
        mul_908: "f32[128]" = torch.ops.aten.mul.Tensor(mul_906, mul_907);  mul_906 = mul_907 = None
        unsqueeze_693: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_908, 0);  mul_908 = None
        unsqueeze_694: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 2);  unsqueeze_693 = None
        unsqueeze_695: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 3);  unsqueeze_694 = None
        mul_909: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_211, primals_212);  primals_212 = None
        unsqueeze_696: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_909, 0);  mul_909 = None
        unsqueeze_697: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, 2);  unsqueeze_696 = None
        unsqueeze_698: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 3);  unsqueeze_697 = None
        mul_910: "f32[32, 128, 5, 5]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_695);  sub_196 = unsqueeze_695 = None
        sub_198: "f32[32, 128, 5, 5]" = torch.ops.aten.sub.Tensor(where_25, mul_910);  where_25 = mul_910 = None
        sub_199: "f32[32, 128, 5, 5]" = torch.ops.aten.sub.Tensor(sub_198, unsqueeze_692);  sub_198 = unsqueeze_692 = None
        mul_911: "f32[32, 128, 5, 5]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_698);  sub_199 = unsqueeze_698 = None
        mul_912: "f32[128]" = torch.ops.aten.mul.Tensor(sum_54, squeeze_211);  sum_54 = squeeze_211 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_911, avg_pool2d_7, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_911 = avg_pool2d_7 = primals_211 = None
        getitem_275: "f32[32, 768, 5, 5]" = convolution_backward_25[0]
        getitem_276: "f32[128, 768, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:382 in forward, code: x = F.avg_pool2d(x, kernel_size=5, stride=3)
        avg_pool2d_backward_2: "f32[32, 768, 17, 17]" = torch.ops.aten.avg_pool2d_backward.default(getitem_275, cat_8, [5, 5], [3, 3], [0, 0], False, True, None);  getitem_275 = cat_8 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:382 in forward, code: x = F.avg_pool2d(x, kernel_size=5, stride=3)
        add_495: "f32[32, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_494, avg_pool2d_backward_2);  add_494 = avg_pool2d_backward_2 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:286 in forward, code: return torch.cat(outputs, 1)
        slice_23: "f32[32, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_495, 1, 0, 192)
        slice_24: "f32[32, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_495, 1, 192, 384)
        slice_25: "f32[32, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_495, 1, 384, 576)
        slice_26: "f32[32, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_495, 1, 576, 768);  add_495 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_26: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_26, full_default, slice_26);  le_26 = slice_26 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_55: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
        sub_200: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_701);  convolution_69 = unsqueeze_701 = None
        mul_913: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_26, sub_200)
        sum_56: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_913, [0, 2, 3]);  mul_913 = None
        mul_914: "f32[192]" = torch.ops.aten.mul.Tensor(sum_55, 0.00010813148788927336)
        unsqueeze_702: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_914, 0);  mul_914 = None
        unsqueeze_703: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 2);  unsqueeze_702 = None
        unsqueeze_704: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 3);  unsqueeze_703 = None
        mul_915: "f32[192]" = torch.ops.aten.mul.Tensor(sum_56, 0.00010813148788927336)
        mul_916: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_208, squeeze_208)
        mul_917: "f32[192]" = torch.ops.aten.mul.Tensor(mul_915, mul_916);  mul_915 = mul_916 = None
        unsqueeze_705: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_917, 0);  mul_917 = None
        unsqueeze_706: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 2);  unsqueeze_705 = None
        unsqueeze_707: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 3);  unsqueeze_706 = None
        mul_918: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_208, primals_209);  primals_209 = None
        unsqueeze_708: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_918, 0);  mul_918 = None
        unsqueeze_709: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, 2);  unsqueeze_708 = None
        unsqueeze_710: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 3);  unsqueeze_709 = None
        mul_919: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_707);  sub_200 = unsqueeze_707 = None
        sub_202: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_26, mul_919);  where_26 = mul_919 = None
        sub_203: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_202, unsqueeze_704);  sub_202 = unsqueeze_704 = None
        mul_920: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_710);  sub_203 = unsqueeze_710 = None
        mul_921: "f32[192]" = torch.ops.aten.mul.Tensor(sum_56, squeeze_208);  sum_56 = squeeze_208 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_920, avg_pool2d_6, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_920 = avg_pool2d_6 = primals_208 = None
        getitem_278: "f32[32, 768, 17, 17]" = convolution_backward_26[0]
        getitem_279: "f32[192, 768, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:278 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward_3: "f32[32, 768, 17, 17]" = torch.ops.aten.avg_pool2d_backward.default(getitem_278, cat_7, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_278 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_27: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_27, full_default, slice_25);  le_27 = slice_25 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_57: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
        sub_204: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_713);  convolution_68 = unsqueeze_713 = None
        mul_922: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_27, sub_204)
        sum_58: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_922, [0, 2, 3]);  mul_922 = None
        mul_923: "f32[192]" = torch.ops.aten.mul.Tensor(sum_57, 0.00010813148788927336)
        unsqueeze_714: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_923, 0);  mul_923 = None
        unsqueeze_715: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 2);  unsqueeze_714 = None
        unsqueeze_716: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 3);  unsqueeze_715 = None
        mul_924: "f32[192]" = torch.ops.aten.mul.Tensor(sum_58, 0.00010813148788927336)
        mul_925: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_205, squeeze_205)
        mul_926: "f32[192]" = torch.ops.aten.mul.Tensor(mul_924, mul_925);  mul_924 = mul_925 = None
        unsqueeze_717: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_926, 0);  mul_926 = None
        unsqueeze_718: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 2);  unsqueeze_717 = None
        unsqueeze_719: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 3);  unsqueeze_718 = None
        mul_927: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_205, primals_206);  primals_206 = None
        unsqueeze_720: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_927, 0);  mul_927 = None
        unsqueeze_721: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, 2);  unsqueeze_720 = None
        unsqueeze_722: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 3);  unsqueeze_721 = None
        mul_928: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_719);  sub_204 = unsqueeze_719 = None
        sub_206: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_27, mul_928);  where_27 = mul_928 = None
        sub_207: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_206, unsqueeze_716);  sub_206 = unsqueeze_716 = None
        mul_929: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_722);  sub_207 = unsqueeze_722 = None
        mul_930: "f32[192]" = torch.ops.aten.mul.Tensor(sum_58, squeeze_205);  sum_58 = squeeze_205 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_929, relu_67, primals_205, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_929 = primals_205 = None
        getitem_281: "f32[32, 192, 17, 17]" = convolution_backward_27[0]
        getitem_282: "f32[192, 192, 1, 7]" = convolution_backward_27[1];  convolution_backward_27 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_306: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_67);  relu_67 = None
        alias_307: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_306);  alias_306 = None
        le_28: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_307, 0);  alias_307 = None
        where_28: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_28, full_default, getitem_281);  le_28 = getitem_281 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_59: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
        sub_208: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_725);  convolution_67 = unsqueeze_725 = None
        mul_931: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_28, sub_208)
        sum_60: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_931, [0, 2, 3]);  mul_931 = None
        mul_932: "f32[192]" = torch.ops.aten.mul.Tensor(sum_59, 0.00010813148788927336)
        unsqueeze_726: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_932, 0);  mul_932 = None
        unsqueeze_727: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 2);  unsqueeze_726 = None
        unsqueeze_728: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 3);  unsqueeze_727 = None
        mul_933: "f32[192]" = torch.ops.aten.mul.Tensor(sum_60, 0.00010813148788927336)
        mul_934: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_202, squeeze_202)
        mul_935: "f32[192]" = torch.ops.aten.mul.Tensor(mul_933, mul_934);  mul_933 = mul_934 = None
        unsqueeze_729: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_935, 0);  mul_935 = None
        unsqueeze_730: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 2);  unsqueeze_729 = None
        unsqueeze_731: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 3);  unsqueeze_730 = None
        mul_936: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_202, primals_203);  primals_203 = None
        unsqueeze_732: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_936, 0);  mul_936 = None
        unsqueeze_733: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, 2);  unsqueeze_732 = None
        unsqueeze_734: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 3);  unsqueeze_733 = None
        mul_937: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_731);  sub_208 = unsqueeze_731 = None
        sub_210: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_28, mul_937);  where_28 = mul_937 = None
        sub_211: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_728);  sub_210 = unsqueeze_728 = None
        mul_938: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_734);  sub_211 = unsqueeze_734 = None
        mul_939: "f32[192]" = torch.ops.aten.mul.Tensor(sum_60, squeeze_202);  sum_60 = squeeze_202 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_938, relu_66, primals_202, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_938 = primals_202 = None
        getitem_284: "f32[32, 192, 17, 17]" = convolution_backward_28[0]
        getitem_285: "f32[192, 192, 7, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_310: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_66);  relu_66 = None
        alias_311: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_310);  alias_310 = None
        le_29: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_311, 0);  alias_311 = None
        where_29: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_29, full_default, getitem_284);  le_29 = getitem_284 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_61: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
        sub_212: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_737);  convolution_66 = unsqueeze_737 = None
        mul_940: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_29, sub_212)
        sum_62: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_940, [0, 2, 3]);  mul_940 = None
        mul_941: "f32[192]" = torch.ops.aten.mul.Tensor(sum_61, 0.00010813148788927336)
        unsqueeze_738: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_941, 0);  mul_941 = None
        unsqueeze_739: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 2);  unsqueeze_738 = None
        unsqueeze_740: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 3);  unsqueeze_739 = None
        mul_942: "f32[192]" = torch.ops.aten.mul.Tensor(sum_62, 0.00010813148788927336)
        mul_943: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
        mul_944: "f32[192]" = torch.ops.aten.mul.Tensor(mul_942, mul_943);  mul_942 = mul_943 = None
        unsqueeze_741: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_944, 0);  mul_944 = None
        unsqueeze_742: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 2);  unsqueeze_741 = None
        unsqueeze_743: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 3);  unsqueeze_742 = None
        mul_945: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_199, primals_200);  primals_200 = None
        unsqueeze_744: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_945, 0);  mul_945 = None
        unsqueeze_745: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, 2);  unsqueeze_744 = None
        unsqueeze_746: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 3);  unsqueeze_745 = None
        mul_946: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_743);  sub_212 = unsqueeze_743 = None
        sub_214: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_29, mul_946);  where_29 = mul_946 = None
        sub_215: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_214, unsqueeze_740);  sub_214 = unsqueeze_740 = None
        mul_947: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_746);  sub_215 = unsqueeze_746 = None
        mul_948: "f32[192]" = torch.ops.aten.mul.Tensor(sum_62, squeeze_199);  sum_62 = squeeze_199 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_947, relu_65, primals_199, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_947 = primals_199 = None
        getitem_287: "f32[32, 192, 17, 17]" = convolution_backward_29[0]
        getitem_288: "f32[192, 192, 1, 7]" = convolution_backward_29[1];  convolution_backward_29 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_314: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_65);  relu_65 = None
        alias_315: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_314);  alias_314 = None
        le_30: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_315, 0);  alias_315 = None
        where_30: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_30, full_default, getitem_287);  le_30 = getitem_287 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_63: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
        sub_216: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_749);  convolution_65 = unsqueeze_749 = None
        mul_949: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_30, sub_216)
        sum_64: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_949, [0, 2, 3]);  mul_949 = None
        mul_950: "f32[192]" = torch.ops.aten.mul.Tensor(sum_63, 0.00010813148788927336)
        unsqueeze_750: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_950, 0);  mul_950 = None
        unsqueeze_751: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 2);  unsqueeze_750 = None
        unsqueeze_752: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 3);  unsqueeze_751 = None
        mul_951: "f32[192]" = torch.ops.aten.mul.Tensor(sum_64, 0.00010813148788927336)
        mul_952: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
        mul_953: "f32[192]" = torch.ops.aten.mul.Tensor(mul_951, mul_952);  mul_951 = mul_952 = None
        unsqueeze_753: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_953, 0);  mul_953 = None
        unsqueeze_754: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 2);  unsqueeze_753 = None
        unsqueeze_755: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 3);  unsqueeze_754 = None
        mul_954: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_196, primals_197);  primals_197 = None
        unsqueeze_756: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_954, 0);  mul_954 = None
        unsqueeze_757: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, 2);  unsqueeze_756 = None
        unsqueeze_758: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 3);  unsqueeze_757 = None
        mul_955: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_755);  sub_216 = unsqueeze_755 = None
        sub_218: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_30, mul_955);  where_30 = mul_955 = None
        sub_219: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_218, unsqueeze_752);  sub_218 = unsqueeze_752 = None
        mul_956: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_758);  sub_219 = unsqueeze_758 = None
        mul_957: "f32[192]" = torch.ops.aten.mul.Tensor(sum_64, squeeze_196);  sum_64 = squeeze_196 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_956, relu_64, primals_196, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_956 = primals_196 = None
        getitem_290: "f32[32, 192, 17, 17]" = convolution_backward_30[0]
        getitem_291: "f32[192, 192, 7, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_318: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_64);  relu_64 = None
        alias_319: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_318);  alias_318 = None
        le_31: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_319, 0);  alias_319 = None
        where_31: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_31, full_default, getitem_290);  le_31 = getitem_290 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_65: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
        sub_220: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_761);  convolution_64 = unsqueeze_761 = None
        mul_958: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_31, sub_220)
        sum_66: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_958, [0, 2, 3]);  mul_958 = None
        mul_959: "f32[192]" = torch.ops.aten.mul.Tensor(sum_65, 0.00010813148788927336)
        unsqueeze_762: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_959, 0);  mul_959 = None
        unsqueeze_763: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 2);  unsqueeze_762 = None
        unsqueeze_764: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 3);  unsqueeze_763 = None
        mul_960: "f32[192]" = torch.ops.aten.mul.Tensor(sum_66, 0.00010813148788927336)
        mul_961: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
        mul_962: "f32[192]" = torch.ops.aten.mul.Tensor(mul_960, mul_961);  mul_960 = mul_961 = None
        unsqueeze_765: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_962, 0);  mul_962 = None
        unsqueeze_766: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 2);  unsqueeze_765 = None
        unsqueeze_767: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 3);  unsqueeze_766 = None
        mul_963: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_193, primals_194);  primals_194 = None
        unsqueeze_768: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_963, 0);  mul_963 = None
        unsqueeze_769: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, 2);  unsqueeze_768 = None
        unsqueeze_770: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 3);  unsqueeze_769 = None
        mul_964: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_767);  sub_220 = unsqueeze_767 = None
        sub_222: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_31, mul_964);  where_31 = mul_964 = None
        sub_223: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_222, unsqueeze_764);  sub_222 = unsqueeze_764 = None
        mul_965: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_770);  sub_223 = unsqueeze_770 = None
        mul_966: "f32[192]" = torch.ops.aten.mul.Tensor(sum_66, squeeze_193);  sum_66 = squeeze_193 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_965, cat_7, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_965 = primals_193 = None
        getitem_293: "f32[32, 768, 17, 17]" = convolution_backward_31[0]
        getitem_294: "f32[192, 768, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_496: "f32[32, 768, 17, 17]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_3, getitem_293);  avg_pool2d_backward_3 = getitem_293 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_32: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_32, full_default, slice_24);  le_32 = slice_24 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_67: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
        sub_224: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_773);  convolution_63 = unsqueeze_773 = None
        mul_967: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_32, sub_224)
        sum_68: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_967, [0, 2, 3]);  mul_967 = None
        mul_968: "f32[192]" = torch.ops.aten.mul.Tensor(sum_67, 0.00010813148788927336)
        unsqueeze_774: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_968, 0);  mul_968 = None
        unsqueeze_775: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 2);  unsqueeze_774 = None
        unsqueeze_776: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 3);  unsqueeze_775 = None
        mul_969: "f32[192]" = torch.ops.aten.mul.Tensor(sum_68, 0.00010813148788927336)
        mul_970: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
        mul_971: "f32[192]" = torch.ops.aten.mul.Tensor(mul_969, mul_970);  mul_969 = mul_970 = None
        unsqueeze_777: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_971, 0);  mul_971 = None
        unsqueeze_778: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 2);  unsqueeze_777 = None
        unsqueeze_779: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 3);  unsqueeze_778 = None
        mul_972: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_191);  primals_191 = None
        unsqueeze_780: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_972, 0);  mul_972 = None
        unsqueeze_781: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, 2);  unsqueeze_780 = None
        unsqueeze_782: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 3);  unsqueeze_781 = None
        mul_973: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_779);  sub_224 = unsqueeze_779 = None
        sub_226: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_32, mul_973);  where_32 = mul_973 = None
        sub_227: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_226, unsqueeze_776);  sub_226 = unsqueeze_776 = None
        mul_974: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_782);  sub_227 = unsqueeze_782 = None
        mul_975: "f32[192]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_190);  sum_68 = squeeze_190 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_974, relu_62, primals_190, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_974 = primals_190 = None
        getitem_296: "f32[32, 192, 17, 17]" = convolution_backward_32[0]
        getitem_297: "f32[192, 192, 7, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_326: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_62);  relu_62 = None
        alias_327: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_326);  alias_326 = None
        le_33: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_327, 0);  alias_327 = None
        where_33: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_33, full_default, getitem_296);  le_33 = getitem_296 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_69: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
        sub_228: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_785);  convolution_62 = unsqueeze_785 = None
        mul_976: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_33, sub_228)
        sum_70: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_976, [0, 2, 3]);  mul_976 = None
        mul_977: "f32[192]" = torch.ops.aten.mul.Tensor(sum_69, 0.00010813148788927336)
        unsqueeze_786: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_977, 0);  mul_977 = None
        unsqueeze_787: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 2);  unsqueeze_786 = None
        unsqueeze_788: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 3);  unsqueeze_787 = None
        mul_978: "f32[192]" = torch.ops.aten.mul.Tensor(sum_70, 0.00010813148788927336)
        mul_979: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
        mul_980: "f32[192]" = torch.ops.aten.mul.Tensor(mul_978, mul_979);  mul_978 = mul_979 = None
        unsqueeze_789: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_980, 0);  mul_980 = None
        unsqueeze_790: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 2);  unsqueeze_789 = None
        unsqueeze_791: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 3);  unsqueeze_790 = None
        mul_981: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_188);  primals_188 = None
        unsqueeze_792: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_981, 0);  mul_981 = None
        unsqueeze_793: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, 2);  unsqueeze_792 = None
        unsqueeze_794: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 3);  unsqueeze_793 = None
        mul_982: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_791);  sub_228 = unsqueeze_791 = None
        sub_230: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_33, mul_982);  where_33 = mul_982 = None
        sub_231: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_230, unsqueeze_788);  sub_230 = unsqueeze_788 = None
        mul_983: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_794);  sub_231 = unsqueeze_794 = None
        mul_984: "f32[192]" = torch.ops.aten.mul.Tensor(sum_70, squeeze_187);  sum_70 = squeeze_187 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_983, relu_61, primals_187, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_983 = primals_187 = None
        getitem_299: "f32[32, 192, 17, 17]" = convolution_backward_33[0]
        getitem_300: "f32[192, 192, 1, 7]" = convolution_backward_33[1];  convolution_backward_33 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_330: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_61);  relu_61 = None
        alias_331: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_330);  alias_330 = None
        le_34: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_331, 0);  alias_331 = None
        where_34: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_34, full_default, getitem_299);  le_34 = getitem_299 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_71: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
        sub_232: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_797);  convolution_61 = unsqueeze_797 = None
        mul_985: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_34, sub_232)
        sum_72: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_985, [0, 2, 3]);  mul_985 = None
        mul_986: "f32[192]" = torch.ops.aten.mul.Tensor(sum_71, 0.00010813148788927336)
        unsqueeze_798: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_986, 0);  mul_986 = None
        unsqueeze_799: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 2);  unsqueeze_798 = None
        unsqueeze_800: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 3);  unsqueeze_799 = None
        mul_987: "f32[192]" = torch.ops.aten.mul.Tensor(sum_72, 0.00010813148788927336)
        mul_988: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
        mul_989: "f32[192]" = torch.ops.aten.mul.Tensor(mul_987, mul_988);  mul_987 = mul_988 = None
        unsqueeze_801: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_989, 0);  mul_989 = None
        unsqueeze_802: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 2);  unsqueeze_801 = None
        unsqueeze_803: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 3);  unsqueeze_802 = None
        mul_990: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_185);  primals_185 = None
        unsqueeze_804: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_990, 0);  mul_990 = None
        unsqueeze_805: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, 2);  unsqueeze_804 = None
        unsqueeze_806: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 3);  unsqueeze_805 = None
        mul_991: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_803);  sub_232 = unsqueeze_803 = None
        sub_234: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_34, mul_991);  where_34 = mul_991 = None
        sub_235: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_234, unsqueeze_800);  sub_234 = unsqueeze_800 = None
        mul_992: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_806);  sub_235 = unsqueeze_806 = None
        mul_993: "f32[192]" = torch.ops.aten.mul.Tensor(sum_72, squeeze_184);  sum_72 = squeeze_184 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_992, cat_7, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_992 = primals_184 = None
        getitem_302: "f32[32, 768, 17, 17]" = convolution_backward_34[0]
        getitem_303: "f32[192, 768, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_497: "f32[32, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_496, getitem_302);  add_496 = getitem_302 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_35: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_35, full_default, slice_23);  le_35 = slice_23 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_73: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
        sub_236: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_809);  convolution_60 = unsqueeze_809 = None
        mul_994: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_35, sub_236)
        sum_74: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_994, [0, 2, 3]);  mul_994 = None
        mul_995: "f32[192]" = torch.ops.aten.mul.Tensor(sum_73, 0.00010813148788927336)
        unsqueeze_810: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_995, 0);  mul_995 = None
        unsqueeze_811: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 2);  unsqueeze_810 = None
        unsqueeze_812: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 3);  unsqueeze_811 = None
        mul_996: "f32[192]" = torch.ops.aten.mul.Tensor(sum_74, 0.00010813148788927336)
        mul_997: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
        mul_998: "f32[192]" = torch.ops.aten.mul.Tensor(mul_996, mul_997);  mul_996 = mul_997 = None
        unsqueeze_813: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_998, 0);  mul_998 = None
        unsqueeze_814: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 2);  unsqueeze_813 = None
        unsqueeze_815: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 3);  unsqueeze_814 = None
        mul_999: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_182);  primals_182 = None
        unsqueeze_816: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_999, 0);  mul_999 = None
        unsqueeze_817: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, 2);  unsqueeze_816 = None
        unsqueeze_818: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 3);  unsqueeze_817 = None
        mul_1000: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_815);  sub_236 = unsqueeze_815 = None
        sub_238: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_35, mul_1000);  where_35 = mul_1000 = None
        sub_239: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_238, unsqueeze_812);  sub_238 = unsqueeze_812 = None
        mul_1001: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_818);  sub_239 = unsqueeze_818 = None
        mul_1002: "f32[192]" = torch.ops.aten.mul.Tensor(sum_74, squeeze_181);  sum_74 = squeeze_181 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_1001, cat_7, primals_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1001 = cat_7 = primals_181 = None
        getitem_305: "f32[32, 768, 17, 17]" = convolution_backward_35[0]
        getitem_306: "f32[192, 768, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_498: "f32[32, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_497, getitem_305);  add_497 = getitem_305 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:286 in forward, code: return torch.cat(outputs, 1)
        slice_27: "f32[32, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_498, 1, 0, 192)
        slice_28: "f32[32, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_498, 1, 192, 384)
        slice_29: "f32[32, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_498, 1, 384, 576)
        slice_30: "f32[32, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_498, 1, 576, 768);  add_498 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_36: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_36, full_default, slice_30);  le_36 = slice_30 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_75: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
        sub_240: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_821);  convolution_59 = unsqueeze_821 = None
        mul_1003: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_36, sub_240)
        sum_76: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1003, [0, 2, 3]);  mul_1003 = None
        mul_1004: "f32[192]" = torch.ops.aten.mul.Tensor(sum_75, 0.00010813148788927336)
        unsqueeze_822: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1004, 0);  mul_1004 = None
        unsqueeze_823: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 2);  unsqueeze_822 = None
        unsqueeze_824: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 3);  unsqueeze_823 = None
        mul_1005: "f32[192]" = torch.ops.aten.mul.Tensor(sum_76, 0.00010813148788927336)
        mul_1006: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
        mul_1007: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1005, mul_1006);  mul_1005 = mul_1006 = None
        unsqueeze_825: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1007, 0);  mul_1007 = None
        unsqueeze_826: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 2);  unsqueeze_825 = None
        unsqueeze_827: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 3);  unsqueeze_826 = None
        mul_1008: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_179);  primals_179 = None
        unsqueeze_828: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1008, 0);  mul_1008 = None
        unsqueeze_829: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, 2);  unsqueeze_828 = None
        unsqueeze_830: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 3);  unsqueeze_829 = None
        mul_1009: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_827);  sub_240 = unsqueeze_827 = None
        sub_242: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_36, mul_1009);  where_36 = mul_1009 = None
        sub_243: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_242, unsqueeze_824);  sub_242 = unsqueeze_824 = None
        mul_1010: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_830);  sub_243 = unsqueeze_830 = None
        mul_1011: "f32[192]" = torch.ops.aten.mul.Tensor(sum_76, squeeze_178);  sum_76 = squeeze_178 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_1010, avg_pool2d_5, primals_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1010 = avg_pool2d_5 = primals_178 = None
        getitem_308: "f32[32, 768, 17, 17]" = convolution_backward_36[0]
        getitem_309: "f32[192, 768, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:278 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward_4: "f32[32, 768, 17, 17]" = torch.ops.aten.avg_pool2d_backward.default(getitem_308, cat_6, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_308 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_37: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_37, full_default, slice_29);  le_37 = slice_29 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_77: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
        sub_244: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_833);  convolution_58 = unsqueeze_833 = None
        mul_1012: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_37, sub_244)
        sum_78: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1012, [0, 2, 3]);  mul_1012 = None
        mul_1013: "f32[192]" = torch.ops.aten.mul.Tensor(sum_77, 0.00010813148788927336)
        unsqueeze_834: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1013, 0);  mul_1013 = None
        unsqueeze_835: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 2);  unsqueeze_834 = None
        unsqueeze_836: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 3);  unsqueeze_835 = None
        mul_1014: "f32[192]" = torch.ops.aten.mul.Tensor(sum_78, 0.00010813148788927336)
        mul_1015: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
        mul_1016: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1014, mul_1015);  mul_1014 = mul_1015 = None
        unsqueeze_837: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1016, 0);  mul_1016 = None
        unsqueeze_838: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 2);  unsqueeze_837 = None
        unsqueeze_839: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 3);  unsqueeze_838 = None
        mul_1017: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_176);  primals_176 = None
        unsqueeze_840: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1017, 0);  mul_1017 = None
        unsqueeze_841: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, 2);  unsqueeze_840 = None
        unsqueeze_842: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 3);  unsqueeze_841 = None
        mul_1018: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_839);  sub_244 = unsqueeze_839 = None
        sub_246: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_37, mul_1018);  where_37 = mul_1018 = None
        sub_247: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_246, unsqueeze_836);  sub_246 = unsqueeze_836 = None
        mul_1019: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_842);  sub_247 = unsqueeze_842 = None
        mul_1020: "f32[192]" = torch.ops.aten.mul.Tensor(sum_78, squeeze_175);  sum_78 = squeeze_175 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_1019, relu_57, primals_175, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1019 = primals_175 = None
        getitem_311: "f32[32, 160, 17, 17]" = convolution_backward_37[0]
        getitem_312: "f32[192, 160, 1, 7]" = convolution_backward_37[1];  convolution_backward_37 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_346: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(relu_57);  relu_57 = None
        alias_347: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(alias_346);  alias_346 = None
        le_38: "b8[32, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_347, 0);  alias_347 = None
        where_38: "f32[32, 160, 17, 17]" = torch.ops.aten.where.self(le_38, full_default, getitem_311);  le_38 = getitem_311 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_79: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
        sub_248: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_845);  convolution_57 = unsqueeze_845 = None
        mul_1021: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_38, sub_248)
        sum_80: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1021, [0, 2, 3]);  mul_1021 = None
        mul_1022: "f32[160]" = torch.ops.aten.mul.Tensor(sum_79, 0.00010813148788927336)
        unsqueeze_846: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1022, 0);  mul_1022 = None
        unsqueeze_847: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 2);  unsqueeze_846 = None
        unsqueeze_848: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 3);  unsqueeze_847 = None
        mul_1023: "f32[160]" = torch.ops.aten.mul.Tensor(sum_80, 0.00010813148788927336)
        mul_1024: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
        mul_1025: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1023, mul_1024);  mul_1023 = mul_1024 = None
        unsqueeze_849: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1025, 0);  mul_1025 = None
        unsqueeze_850: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 2);  unsqueeze_849 = None
        unsqueeze_851: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 3);  unsqueeze_850 = None
        mul_1026: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_173);  primals_173 = None
        unsqueeze_852: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1026, 0);  mul_1026 = None
        unsqueeze_853: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, 2);  unsqueeze_852 = None
        unsqueeze_854: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 3);  unsqueeze_853 = None
        mul_1027: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_851);  sub_248 = unsqueeze_851 = None
        sub_250: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_38, mul_1027);  where_38 = mul_1027 = None
        sub_251: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_250, unsqueeze_848);  sub_250 = unsqueeze_848 = None
        mul_1028: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_854);  sub_251 = unsqueeze_854 = None
        mul_1029: "f32[160]" = torch.ops.aten.mul.Tensor(sum_80, squeeze_172);  sum_80 = squeeze_172 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_1028, relu_56, primals_172, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1028 = primals_172 = None
        getitem_314: "f32[32, 160, 17, 17]" = convolution_backward_38[0]
        getitem_315: "f32[160, 160, 7, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_350: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(relu_56);  relu_56 = None
        alias_351: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(alias_350);  alias_350 = None
        le_39: "b8[32, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_351, 0);  alias_351 = None
        where_39: "f32[32, 160, 17, 17]" = torch.ops.aten.where.self(le_39, full_default, getitem_314);  le_39 = getitem_314 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_81: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
        sub_252: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_857);  convolution_56 = unsqueeze_857 = None
        mul_1030: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_39, sub_252)
        sum_82: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1030, [0, 2, 3]);  mul_1030 = None
        mul_1031: "f32[160]" = torch.ops.aten.mul.Tensor(sum_81, 0.00010813148788927336)
        unsqueeze_858: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1031, 0);  mul_1031 = None
        unsqueeze_859: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 2);  unsqueeze_858 = None
        unsqueeze_860: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 3);  unsqueeze_859 = None
        mul_1032: "f32[160]" = torch.ops.aten.mul.Tensor(sum_82, 0.00010813148788927336)
        mul_1033: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
        mul_1034: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1032, mul_1033);  mul_1032 = mul_1033 = None
        unsqueeze_861: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1034, 0);  mul_1034 = None
        unsqueeze_862: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 2);  unsqueeze_861 = None
        unsqueeze_863: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 3);  unsqueeze_862 = None
        mul_1035: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_170);  primals_170 = None
        unsqueeze_864: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1035, 0);  mul_1035 = None
        unsqueeze_865: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, 2);  unsqueeze_864 = None
        unsqueeze_866: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 3);  unsqueeze_865 = None
        mul_1036: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_863);  sub_252 = unsqueeze_863 = None
        sub_254: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_39, mul_1036);  where_39 = mul_1036 = None
        sub_255: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_254, unsqueeze_860);  sub_254 = unsqueeze_860 = None
        mul_1037: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_866);  sub_255 = unsqueeze_866 = None
        mul_1038: "f32[160]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_169);  sum_82 = squeeze_169 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_1037, relu_55, primals_169, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1037 = primals_169 = None
        getitem_317: "f32[32, 160, 17, 17]" = convolution_backward_39[0]
        getitem_318: "f32[160, 160, 1, 7]" = convolution_backward_39[1];  convolution_backward_39 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_354: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(relu_55);  relu_55 = None
        alias_355: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(alias_354);  alias_354 = None
        le_40: "b8[32, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_355, 0);  alias_355 = None
        where_40: "f32[32, 160, 17, 17]" = torch.ops.aten.where.self(le_40, full_default, getitem_317);  le_40 = getitem_317 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_83: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
        sub_256: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_869);  convolution_55 = unsqueeze_869 = None
        mul_1039: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_40, sub_256)
        sum_84: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1039, [0, 2, 3]);  mul_1039 = None
        mul_1040: "f32[160]" = torch.ops.aten.mul.Tensor(sum_83, 0.00010813148788927336)
        unsqueeze_870: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1040, 0);  mul_1040 = None
        unsqueeze_871: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 2);  unsqueeze_870 = None
        unsqueeze_872: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 3);  unsqueeze_871 = None
        mul_1041: "f32[160]" = torch.ops.aten.mul.Tensor(sum_84, 0.00010813148788927336)
        mul_1042: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
        mul_1043: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1041, mul_1042);  mul_1041 = mul_1042 = None
        unsqueeze_873: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1043, 0);  mul_1043 = None
        unsqueeze_874: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 2);  unsqueeze_873 = None
        unsqueeze_875: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 3);  unsqueeze_874 = None
        mul_1044: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_167);  primals_167 = None
        unsqueeze_876: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1044, 0);  mul_1044 = None
        unsqueeze_877: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, 2);  unsqueeze_876 = None
        unsqueeze_878: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 3);  unsqueeze_877 = None
        mul_1045: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_875);  sub_256 = unsqueeze_875 = None
        sub_258: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_40, mul_1045);  where_40 = mul_1045 = None
        sub_259: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_258, unsqueeze_872);  sub_258 = unsqueeze_872 = None
        mul_1046: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_878);  sub_259 = unsqueeze_878 = None
        mul_1047: "f32[160]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_166);  sum_84 = squeeze_166 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_1046, relu_54, primals_166, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1046 = primals_166 = None
        getitem_320: "f32[32, 160, 17, 17]" = convolution_backward_40[0]
        getitem_321: "f32[160, 160, 7, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_358: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(relu_54);  relu_54 = None
        alias_359: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(alias_358);  alias_358 = None
        le_41: "b8[32, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_359, 0);  alias_359 = None
        where_41: "f32[32, 160, 17, 17]" = torch.ops.aten.where.self(le_41, full_default, getitem_320);  le_41 = getitem_320 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_85: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
        sub_260: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_881);  convolution_54 = unsqueeze_881 = None
        mul_1048: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_41, sub_260)
        sum_86: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1048, [0, 2, 3]);  mul_1048 = None
        mul_1049: "f32[160]" = torch.ops.aten.mul.Tensor(sum_85, 0.00010813148788927336)
        unsqueeze_882: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1049, 0);  mul_1049 = None
        unsqueeze_883: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 2);  unsqueeze_882 = None
        unsqueeze_884: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 3);  unsqueeze_883 = None
        mul_1050: "f32[160]" = torch.ops.aten.mul.Tensor(sum_86, 0.00010813148788927336)
        mul_1051: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
        mul_1052: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1050, mul_1051);  mul_1050 = mul_1051 = None
        unsqueeze_885: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1052, 0);  mul_1052 = None
        unsqueeze_886: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 2);  unsqueeze_885 = None
        unsqueeze_887: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 3);  unsqueeze_886 = None
        mul_1053: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_164);  primals_164 = None
        unsqueeze_888: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1053, 0);  mul_1053 = None
        unsqueeze_889: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, 2);  unsqueeze_888 = None
        unsqueeze_890: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 3);  unsqueeze_889 = None
        mul_1054: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_887);  sub_260 = unsqueeze_887 = None
        sub_262: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_41, mul_1054);  where_41 = mul_1054 = None
        sub_263: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_262, unsqueeze_884);  sub_262 = unsqueeze_884 = None
        mul_1055: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_890);  sub_263 = unsqueeze_890 = None
        mul_1056: "f32[160]" = torch.ops.aten.mul.Tensor(sum_86, squeeze_163);  sum_86 = squeeze_163 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_1055, cat_6, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1055 = primals_163 = None
        getitem_323: "f32[32, 768, 17, 17]" = convolution_backward_41[0]
        getitem_324: "f32[160, 768, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_499: "f32[32, 768, 17, 17]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_4, getitem_323);  avg_pool2d_backward_4 = getitem_323 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_42: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_42, full_default, slice_28);  le_42 = slice_28 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_87: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
        sub_264: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_893);  convolution_53 = unsqueeze_893 = None
        mul_1057: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_42, sub_264)
        sum_88: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1057, [0, 2, 3]);  mul_1057 = None
        mul_1058: "f32[192]" = torch.ops.aten.mul.Tensor(sum_87, 0.00010813148788927336)
        unsqueeze_894: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1058, 0);  mul_1058 = None
        unsqueeze_895: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 2);  unsqueeze_894 = None
        unsqueeze_896: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 3);  unsqueeze_895 = None
        mul_1059: "f32[192]" = torch.ops.aten.mul.Tensor(sum_88, 0.00010813148788927336)
        mul_1060: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
        mul_1061: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1059, mul_1060);  mul_1059 = mul_1060 = None
        unsqueeze_897: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1061, 0);  mul_1061 = None
        unsqueeze_898: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 2);  unsqueeze_897 = None
        unsqueeze_899: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 3);  unsqueeze_898 = None
        mul_1062: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_161);  primals_161 = None
        unsqueeze_900: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1062, 0);  mul_1062 = None
        unsqueeze_901: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, 2);  unsqueeze_900 = None
        unsqueeze_902: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 3);  unsqueeze_901 = None
        mul_1063: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_899);  sub_264 = unsqueeze_899 = None
        sub_266: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_42, mul_1063);  where_42 = mul_1063 = None
        sub_267: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_266, unsqueeze_896);  sub_266 = unsqueeze_896 = None
        mul_1064: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_902);  sub_267 = unsqueeze_902 = None
        mul_1065: "f32[192]" = torch.ops.aten.mul.Tensor(sum_88, squeeze_160);  sum_88 = squeeze_160 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_1064, relu_52, primals_160, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1064 = primals_160 = None
        getitem_326: "f32[32, 160, 17, 17]" = convolution_backward_42[0]
        getitem_327: "f32[192, 160, 7, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_366: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(relu_52);  relu_52 = None
        alias_367: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(alias_366);  alias_366 = None
        le_43: "b8[32, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_367, 0);  alias_367 = None
        where_43: "f32[32, 160, 17, 17]" = torch.ops.aten.where.self(le_43, full_default, getitem_326);  le_43 = getitem_326 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_89: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
        sub_268: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_905);  convolution_52 = unsqueeze_905 = None
        mul_1066: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_43, sub_268)
        sum_90: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1066, [0, 2, 3]);  mul_1066 = None
        mul_1067: "f32[160]" = torch.ops.aten.mul.Tensor(sum_89, 0.00010813148788927336)
        unsqueeze_906: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1067, 0);  mul_1067 = None
        unsqueeze_907: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 2);  unsqueeze_906 = None
        unsqueeze_908: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 3);  unsqueeze_907 = None
        mul_1068: "f32[160]" = torch.ops.aten.mul.Tensor(sum_90, 0.00010813148788927336)
        mul_1069: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
        mul_1070: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1068, mul_1069);  mul_1068 = mul_1069 = None
        unsqueeze_909: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1070, 0);  mul_1070 = None
        unsqueeze_910: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 2);  unsqueeze_909 = None
        unsqueeze_911: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 3);  unsqueeze_910 = None
        mul_1071: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_158);  primals_158 = None
        unsqueeze_912: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1071, 0);  mul_1071 = None
        unsqueeze_913: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, 2);  unsqueeze_912 = None
        unsqueeze_914: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 3);  unsqueeze_913 = None
        mul_1072: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_911);  sub_268 = unsqueeze_911 = None
        sub_270: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_43, mul_1072);  where_43 = mul_1072 = None
        sub_271: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_270, unsqueeze_908);  sub_270 = unsqueeze_908 = None
        mul_1073: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_271, unsqueeze_914);  sub_271 = unsqueeze_914 = None
        mul_1074: "f32[160]" = torch.ops.aten.mul.Tensor(sum_90, squeeze_157);  sum_90 = squeeze_157 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_1073, relu_51, primals_157, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1073 = primals_157 = None
        getitem_329: "f32[32, 160, 17, 17]" = convolution_backward_43[0]
        getitem_330: "f32[160, 160, 1, 7]" = convolution_backward_43[1];  convolution_backward_43 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_370: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(relu_51);  relu_51 = None
        alias_371: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(alias_370);  alias_370 = None
        le_44: "b8[32, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_371, 0);  alias_371 = None
        where_44: "f32[32, 160, 17, 17]" = torch.ops.aten.where.self(le_44, full_default, getitem_329);  le_44 = getitem_329 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_91: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
        sub_272: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_917);  convolution_51 = unsqueeze_917 = None
        mul_1075: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_44, sub_272)
        sum_92: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1075, [0, 2, 3]);  mul_1075 = None
        mul_1076: "f32[160]" = torch.ops.aten.mul.Tensor(sum_91, 0.00010813148788927336)
        unsqueeze_918: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1076, 0);  mul_1076 = None
        unsqueeze_919: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 2);  unsqueeze_918 = None
        unsqueeze_920: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_919, 3);  unsqueeze_919 = None
        mul_1077: "f32[160]" = torch.ops.aten.mul.Tensor(sum_92, 0.00010813148788927336)
        mul_1078: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
        mul_1079: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1077, mul_1078);  mul_1077 = mul_1078 = None
        unsqueeze_921: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1079, 0);  mul_1079 = None
        unsqueeze_922: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_921, 2);  unsqueeze_921 = None
        unsqueeze_923: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, 3);  unsqueeze_922 = None
        mul_1080: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_155);  primals_155 = None
        unsqueeze_924: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1080, 0);  mul_1080 = None
        unsqueeze_925: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, 2);  unsqueeze_924 = None
        unsqueeze_926: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 3);  unsqueeze_925 = None
        mul_1081: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_923);  sub_272 = unsqueeze_923 = None
        sub_274: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_44, mul_1081);  where_44 = mul_1081 = None
        sub_275: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_274, unsqueeze_920);  sub_274 = unsqueeze_920 = None
        mul_1082: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_926);  sub_275 = unsqueeze_926 = None
        mul_1083: "f32[160]" = torch.ops.aten.mul.Tensor(sum_92, squeeze_154);  sum_92 = squeeze_154 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_1082, cat_6, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1082 = primals_154 = None
        getitem_332: "f32[32, 768, 17, 17]" = convolution_backward_44[0]
        getitem_333: "f32[160, 768, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_500: "f32[32, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_499, getitem_332);  add_499 = getitem_332 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_45: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_45, full_default, slice_27);  le_45 = slice_27 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_93: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
        sub_276: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_929);  convolution_50 = unsqueeze_929 = None
        mul_1084: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_45, sub_276)
        sum_94: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1084, [0, 2, 3]);  mul_1084 = None
        mul_1085: "f32[192]" = torch.ops.aten.mul.Tensor(sum_93, 0.00010813148788927336)
        unsqueeze_930: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1085, 0);  mul_1085 = None
        unsqueeze_931: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 2);  unsqueeze_930 = None
        unsqueeze_932: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_931, 3);  unsqueeze_931 = None
        mul_1086: "f32[192]" = torch.ops.aten.mul.Tensor(sum_94, 0.00010813148788927336)
        mul_1087: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
        mul_1088: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1086, mul_1087);  mul_1086 = mul_1087 = None
        unsqueeze_933: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1088, 0);  mul_1088 = None
        unsqueeze_934: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_933, 2);  unsqueeze_933 = None
        unsqueeze_935: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, 3);  unsqueeze_934 = None
        mul_1089: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_152);  primals_152 = None
        unsqueeze_936: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1089, 0);  mul_1089 = None
        unsqueeze_937: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, 2);  unsqueeze_936 = None
        unsqueeze_938: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_937, 3);  unsqueeze_937 = None
        mul_1090: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_935);  sub_276 = unsqueeze_935 = None
        sub_278: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_45, mul_1090);  where_45 = mul_1090 = None
        sub_279: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_278, unsqueeze_932);  sub_278 = unsqueeze_932 = None
        mul_1091: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_279, unsqueeze_938);  sub_279 = unsqueeze_938 = None
        mul_1092: "f32[192]" = torch.ops.aten.mul.Tensor(sum_94, squeeze_151);  sum_94 = squeeze_151 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_1091, cat_6, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1091 = cat_6 = primals_151 = None
        getitem_335: "f32[32, 768, 17, 17]" = convolution_backward_45[0]
        getitem_336: "f32[192, 768, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_501: "f32[32, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_500, getitem_335);  add_500 = getitem_335 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:286 in forward, code: return torch.cat(outputs, 1)
        slice_31: "f32[32, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_501, 1, 0, 192)
        slice_32: "f32[32, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_501, 1, 192, 384)
        slice_33: "f32[32, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_501, 1, 384, 576)
        slice_34: "f32[32, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_501, 1, 576, 768);  add_501 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_46: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_46, full_default, slice_34);  le_46 = slice_34 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_95: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
        sub_280: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_941);  convolution_49 = unsqueeze_941 = None
        mul_1093: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_46, sub_280)
        sum_96: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1093, [0, 2, 3]);  mul_1093 = None
        mul_1094: "f32[192]" = torch.ops.aten.mul.Tensor(sum_95, 0.00010813148788927336)
        unsqueeze_942: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1094, 0);  mul_1094 = None
        unsqueeze_943: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 2);  unsqueeze_942 = None
        unsqueeze_944: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_943, 3);  unsqueeze_943 = None
        mul_1095: "f32[192]" = torch.ops.aten.mul.Tensor(sum_96, 0.00010813148788927336)
        mul_1096: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
        mul_1097: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1095, mul_1096);  mul_1095 = mul_1096 = None
        unsqueeze_945: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1097, 0);  mul_1097 = None
        unsqueeze_946: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_945, 2);  unsqueeze_945 = None
        unsqueeze_947: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, 3);  unsqueeze_946 = None
        mul_1098: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_149);  primals_149 = None
        unsqueeze_948: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1098, 0);  mul_1098 = None
        unsqueeze_949: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, 2);  unsqueeze_948 = None
        unsqueeze_950: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_949, 3);  unsqueeze_949 = None
        mul_1099: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_947);  sub_280 = unsqueeze_947 = None
        sub_282: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_46, mul_1099);  where_46 = mul_1099 = None
        sub_283: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_282, unsqueeze_944);  sub_282 = unsqueeze_944 = None
        mul_1100: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_283, unsqueeze_950);  sub_283 = unsqueeze_950 = None
        mul_1101: "f32[192]" = torch.ops.aten.mul.Tensor(sum_96, squeeze_148);  sum_96 = squeeze_148 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_1100, avg_pool2d_4, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1100 = avg_pool2d_4 = primals_148 = None
        getitem_338: "f32[32, 768, 17, 17]" = convolution_backward_46[0]
        getitem_339: "f32[192, 768, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:278 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward_5: "f32[32, 768, 17, 17]" = torch.ops.aten.avg_pool2d_backward.default(getitem_338, cat_5, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_338 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_47: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_47, full_default, slice_33);  le_47 = slice_33 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_97: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
        sub_284: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_953);  convolution_48 = unsqueeze_953 = None
        mul_1102: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_47, sub_284)
        sum_98: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1102, [0, 2, 3]);  mul_1102 = None
        mul_1103: "f32[192]" = torch.ops.aten.mul.Tensor(sum_97, 0.00010813148788927336)
        unsqueeze_954: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1103, 0);  mul_1103 = None
        unsqueeze_955: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 2);  unsqueeze_954 = None
        unsqueeze_956: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_955, 3);  unsqueeze_955 = None
        mul_1104: "f32[192]" = torch.ops.aten.mul.Tensor(sum_98, 0.00010813148788927336)
        mul_1105: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
        mul_1106: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1104, mul_1105);  mul_1104 = mul_1105 = None
        unsqueeze_957: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1106, 0);  mul_1106 = None
        unsqueeze_958: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_957, 2);  unsqueeze_957 = None
        unsqueeze_959: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, 3);  unsqueeze_958 = None
        mul_1107: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_146);  primals_146 = None
        unsqueeze_960: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1107, 0);  mul_1107 = None
        unsqueeze_961: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, 2);  unsqueeze_960 = None
        unsqueeze_962: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_961, 3);  unsqueeze_961 = None
        mul_1108: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_959);  sub_284 = unsqueeze_959 = None
        sub_286: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_47, mul_1108);  where_47 = mul_1108 = None
        sub_287: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_286, unsqueeze_956);  sub_286 = unsqueeze_956 = None
        mul_1109: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_287, unsqueeze_962);  sub_287 = unsqueeze_962 = None
        mul_1110: "f32[192]" = torch.ops.aten.mul.Tensor(sum_98, squeeze_145);  sum_98 = squeeze_145 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_1109, relu_47, primals_145, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1109 = primals_145 = None
        getitem_341: "f32[32, 160, 17, 17]" = convolution_backward_47[0]
        getitem_342: "f32[192, 160, 1, 7]" = convolution_backward_47[1];  convolution_backward_47 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_386: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(relu_47);  relu_47 = None
        alias_387: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(alias_386);  alias_386 = None
        le_48: "b8[32, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_387, 0);  alias_387 = None
        where_48: "f32[32, 160, 17, 17]" = torch.ops.aten.where.self(le_48, full_default, getitem_341);  le_48 = getitem_341 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_99: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
        sub_288: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_965);  convolution_47 = unsqueeze_965 = None
        mul_1111: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_48, sub_288)
        sum_100: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1111, [0, 2, 3]);  mul_1111 = None
        mul_1112: "f32[160]" = torch.ops.aten.mul.Tensor(sum_99, 0.00010813148788927336)
        unsqueeze_966: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1112, 0);  mul_1112 = None
        unsqueeze_967: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 2);  unsqueeze_966 = None
        unsqueeze_968: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_967, 3);  unsqueeze_967 = None
        mul_1113: "f32[160]" = torch.ops.aten.mul.Tensor(sum_100, 0.00010813148788927336)
        mul_1114: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_1115: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1113, mul_1114);  mul_1113 = mul_1114 = None
        unsqueeze_969: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1115, 0);  mul_1115 = None
        unsqueeze_970: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_969, 2);  unsqueeze_969 = None
        unsqueeze_971: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, 3);  unsqueeze_970 = None
        mul_1116: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_143);  primals_143 = None
        unsqueeze_972: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1116, 0);  mul_1116 = None
        unsqueeze_973: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, 2);  unsqueeze_972 = None
        unsqueeze_974: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_973, 3);  unsqueeze_973 = None
        mul_1117: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_971);  sub_288 = unsqueeze_971 = None
        sub_290: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_48, mul_1117);  where_48 = mul_1117 = None
        sub_291: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_290, unsqueeze_968);  sub_290 = unsqueeze_968 = None
        mul_1118: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_974);  sub_291 = unsqueeze_974 = None
        mul_1119: "f32[160]" = torch.ops.aten.mul.Tensor(sum_100, squeeze_142);  sum_100 = squeeze_142 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_1118, relu_46, primals_142, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1118 = primals_142 = None
        getitem_344: "f32[32, 160, 17, 17]" = convolution_backward_48[0]
        getitem_345: "f32[160, 160, 7, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_390: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(relu_46);  relu_46 = None
        alias_391: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(alias_390);  alias_390 = None
        le_49: "b8[32, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_391, 0);  alias_391 = None
        where_49: "f32[32, 160, 17, 17]" = torch.ops.aten.where.self(le_49, full_default, getitem_344);  le_49 = getitem_344 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_101: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
        sub_292: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_977);  convolution_46 = unsqueeze_977 = None
        mul_1120: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_49, sub_292)
        sum_102: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1120, [0, 2, 3]);  mul_1120 = None
        mul_1121: "f32[160]" = torch.ops.aten.mul.Tensor(sum_101, 0.00010813148788927336)
        unsqueeze_978: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1121, 0);  mul_1121 = None
        unsqueeze_979: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 2);  unsqueeze_978 = None
        unsqueeze_980: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_979, 3);  unsqueeze_979 = None
        mul_1122: "f32[160]" = torch.ops.aten.mul.Tensor(sum_102, 0.00010813148788927336)
        mul_1123: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
        mul_1124: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1122, mul_1123);  mul_1122 = mul_1123 = None
        unsqueeze_981: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1124, 0);  mul_1124 = None
        unsqueeze_982: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_981, 2);  unsqueeze_981 = None
        unsqueeze_983: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, 3);  unsqueeze_982 = None
        mul_1125: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_140);  primals_140 = None
        unsqueeze_984: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1125, 0);  mul_1125 = None
        unsqueeze_985: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, 2);  unsqueeze_984 = None
        unsqueeze_986: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_985, 3);  unsqueeze_985 = None
        mul_1126: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_983);  sub_292 = unsqueeze_983 = None
        sub_294: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_49, mul_1126);  where_49 = mul_1126 = None
        sub_295: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_294, unsqueeze_980);  sub_294 = unsqueeze_980 = None
        mul_1127: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_295, unsqueeze_986);  sub_295 = unsqueeze_986 = None
        mul_1128: "f32[160]" = torch.ops.aten.mul.Tensor(sum_102, squeeze_139);  sum_102 = squeeze_139 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_1127, relu_45, primals_139, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1127 = primals_139 = None
        getitem_347: "f32[32, 160, 17, 17]" = convolution_backward_49[0]
        getitem_348: "f32[160, 160, 1, 7]" = convolution_backward_49[1];  convolution_backward_49 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_394: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(relu_45);  relu_45 = None
        alias_395: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(alias_394);  alias_394 = None
        le_50: "b8[32, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_395, 0);  alias_395 = None
        where_50: "f32[32, 160, 17, 17]" = torch.ops.aten.where.self(le_50, full_default, getitem_347);  le_50 = getitem_347 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_103: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
        sub_296: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_989);  convolution_45 = unsqueeze_989 = None
        mul_1129: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_50, sub_296)
        sum_104: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1129, [0, 2, 3]);  mul_1129 = None
        mul_1130: "f32[160]" = torch.ops.aten.mul.Tensor(sum_103, 0.00010813148788927336)
        unsqueeze_990: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1130, 0);  mul_1130 = None
        unsqueeze_991: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 2);  unsqueeze_990 = None
        unsqueeze_992: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_991, 3);  unsqueeze_991 = None
        mul_1131: "f32[160]" = torch.ops.aten.mul.Tensor(sum_104, 0.00010813148788927336)
        mul_1132: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
        mul_1133: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1131, mul_1132);  mul_1131 = mul_1132 = None
        unsqueeze_993: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1133, 0);  mul_1133 = None
        unsqueeze_994: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_993, 2);  unsqueeze_993 = None
        unsqueeze_995: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_994, 3);  unsqueeze_994 = None
        mul_1134: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_137);  primals_137 = None
        unsqueeze_996: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1134, 0);  mul_1134 = None
        unsqueeze_997: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_996, 2);  unsqueeze_996 = None
        unsqueeze_998: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_997, 3);  unsqueeze_997 = None
        mul_1135: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_995);  sub_296 = unsqueeze_995 = None
        sub_298: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_50, mul_1135);  where_50 = mul_1135 = None
        sub_299: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_298, unsqueeze_992);  sub_298 = unsqueeze_992 = None
        mul_1136: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_998);  sub_299 = unsqueeze_998 = None
        mul_1137: "f32[160]" = torch.ops.aten.mul.Tensor(sum_104, squeeze_136);  sum_104 = squeeze_136 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_1136, relu_44, primals_136, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1136 = primals_136 = None
        getitem_350: "f32[32, 160, 17, 17]" = convolution_backward_50[0]
        getitem_351: "f32[160, 160, 7, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_398: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(relu_44);  relu_44 = None
        alias_399: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(alias_398);  alias_398 = None
        le_51: "b8[32, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_399, 0);  alias_399 = None
        where_51: "f32[32, 160, 17, 17]" = torch.ops.aten.where.self(le_51, full_default, getitem_350);  le_51 = getitem_350 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_105: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
        sub_300: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_1001);  convolution_44 = unsqueeze_1001 = None
        mul_1138: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_51, sub_300)
        sum_106: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1138, [0, 2, 3]);  mul_1138 = None
        mul_1139: "f32[160]" = torch.ops.aten.mul.Tensor(sum_105, 0.00010813148788927336)
        unsqueeze_1002: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1139, 0);  mul_1139 = None
        unsqueeze_1003: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, 2);  unsqueeze_1002 = None
        unsqueeze_1004: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1003, 3);  unsqueeze_1003 = None
        mul_1140: "f32[160]" = torch.ops.aten.mul.Tensor(sum_106, 0.00010813148788927336)
        mul_1141: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
        mul_1142: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1140, mul_1141);  mul_1140 = mul_1141 = None
        unsqueeze_1005: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1142, 0);  mul_1142 = None
        unsqueeze_1006: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1005, 2);  unsqueeze_1005 = None
        unsqueeze_1007: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, 3);  unsqueeze_1006 = None
        mul_1143: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_134);  primals_134 = None
        unsqueeze_1008: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1143, 0);  mul_1143 = None
        unsqueeze_1009: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, 2);  unsqueeze_1008 = None
        unsqueeze_1010: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1009, 3);  unsqueeze_1009 = None
        mul_1144: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_1007);  sub_300 = unsqueeze_1007 = None
        sub_302: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_51, mul_1144);  where_51 = mul_1144 = None
        sub_303: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_302, unsqueeze_1004);  sub_302 = unsqueeze_1004 = None
        mul_1145: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_1010);  sub_303 = unsqueeze_1010 = None
        mul_1146: "f32[160]" = torch.ops.aten.mul.Tensor(sum_106, squeeze_133);  sum_106 = squeeze_133 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_1145, cat_5, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1145 = primals_133 = None
        getitem_353: "f32[32, 768, 17, 17]" = convolution_backward_51[0]
        getitem_354: "f32[160, 768, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_502: "f32[32, 768, 17, 17]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_5, getitem_353);  avg_pool2d_backward_5 = getitem_353 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_52: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_52, full_default, slice_32);  le_52 = slice_32 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_107: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
        sub_304: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_1013);  convolution_43 = unsqueeze_1013 = None
        mul_1147: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_52, sub_304)
        sum_108: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1147, [0, 2, 3]);  mul_1147 = None
        mul_1148: "f32[192]" = torch.ops.aten.mul.Tensor(sum_107, 0.00010813148788927336)
        unsqueeze_1014: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1148, 0);  mul_1148 = None
        unsqueeze_1015: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, 2);  unsqueeze_1014 = None
        unsqueeze_1016: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1015, 3);  unsqueeze_1015 = None
        mul_1149: "f32[192]" = torch.ops.aten.mul.Tensor(sum_108, 0.00010813148788927336)
        mul_1150: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
        mul_1151: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1149, mul_1150);  mul_1149 = mul_1150 = None
        unsqueeze_1017: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1151, 0);  mul_1151 = None
        unsqueeze_1018: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1017, 2);  unsqueeze_1017 = None
        unsqueeze_1019: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, 3);  unsqueeze_1018 = None
        mul_1152: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_131);  primals_131 = None
        unsqueeze_1020: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1152, 0);  mul_1152 = None
        unsqueeze_1021: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1020, 2);  unsqueeze_1020 = None
        unsqueeze_1022: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1021, 3);  unsqueeze_1021 = None
        mul_1153: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_1019);  sub_304 = unsqueeze_1019 = None
        sub_306: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_52, mul_1153);  where_52 = mul_1153 = None
        sub_307: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_306, unsqueeze_1016);  sub_306 = unsqueeze_1016 = None
        mul_1154: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_307, unsqueeze_1022);  sub_307 = unsqueeze_1022 = None
        mul_1155: "f32[192]" = torch.ops.aten.mul.Tensor(sum_108, squeeze_130);  sum_108 = squeeze_130 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_1154, relu_42, primals_130, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1154 = primals_130 = None
        getitem_356: "f32[32, 160, 17, 17]" = convolution_backward_52[0]
        getitem_357: "f32[192, 160, 7, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_406: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(relu_42);  relu_42 = None
        alias_407: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(alias_406);  alias_406 = None
        le_53: "b8[32, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_407, 0);  alias_407 = None
        where_53: "f32[32, 160, 17, 17]" = torch.ops.aten.where.self(le_53, full_default, getitem_356);  le_53 = getitem_356 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_109: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
        sub_308: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_1025);  convolution_42 = unsqueeze_1025 = None
        mul_1156: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_53, sub_308)
        sum_110: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1156, [0, 2, 3]);  mul_1156 = None
        mul_1157: "f32[160]" = torch.ops.aten.mul.Tensor(sum_109, 0.00010813148788927336)
        unsqueeze_1026: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1157, 0);  mul_1157 = None
        unsqueeze_1027: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, 2);  unsqueeze_1026 = None
        unsqueeze_1028: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1027, 3);  unsqueeze_1027 = None
        mul_1158: "f32[160]" = torch.ops.aten.mul.Tensor(sum_110, 0.00010813148788927336)
        mul_1159: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
        mul_1160: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1158, mul_1159);  mul_1158 = mul_1159 = None
        unsqueeze_1029: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1160, 0);  mul_1160 = None
        unsqueeze_1030: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1029, 2);  unsqueeze_1029 = None
        unsqueeze_1031: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, 3);  unsqueeze_1030 = None
        mul_1161: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_128);  primals_128 = None
        unsqueeze_1032: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1161, 0);  mul_1161 = None
        unsqueeze_1033: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, 2);  unsqueeze_1032 = None
        unsqueeze_1034: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1033, 3);  unsqueeze_1033 = None
        mul_1162: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_1031);  sub_308 = unsqueeze_1031 = None
        sub_310: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_53, mul_1162);  where_53 = mul_1162 = None
        sub_311: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_310, unsqueeze_1028);  sub_310 = unsqueeze_1028 = None
        mul_1163: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_311, unsqueeze_1034);  sub_311 = unsqueeze_1034 = None
        mul_1164: "f32[160]" = torch.ops.aten.mul.Tensor(sum_110, squeeze_127);  sum_110 = squeeze_127 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_1163, relu_41, primals_127, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1163 = primals_127 = None
        getitem_359: "f32[32, 160, 17, 17]" = convolution_backward_53[0]
        getitem_360: "f32[160, 160, 1, 7]" = convolution_backward_53[1];  convolution_backward_53 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_410: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(relu_41);  relu_41 = None
        alias_411: "f32[32, 160, 17, 17]" = torch.ops.aten.alias.default(alias_410);  alias_410 = None
        le_54: "b8[32, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_411, 0);  alias_411 = None
        where_54: "f32[32, 160, 17, 17]" = torch.ops.aten.where.self(le_54, full_default, getitem_359);  le_54 = getitem_359 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_111: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
        sub_312: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_1037);  convolution_41 = unsqueeze_1037 = None
        mul_1165: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_54, sub_312)
        sum_112: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1165, [0, 2, 3]);  mul_1165 = None
        mul_1166: "f32[160]" = torch.ops.aten.mul.Tensor(sum_111, 0.00010813148788927336)
        unsqueeze_1038: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1166, 0);  mul_1166 = None
        unsqueeze_1039: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 2);  unsqueeze_1038 = None
        unsqueeze_1040: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1039, 3);  unsqueeze_1039 = None
        mul_1167: "f32[160]" = torch.ops.aten.mul.Tensor(sum_112, 0.00010813148788927336)
        mul_1168: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_1169: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1167, mul_1168);  mul_1167 = mul_1168 = None
        unsqueeze_1041: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1169, 0);  mul_1169 = None
        unsqueeze_1042: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1041, 2);  unsqueeze_1041 = None
        unsqueeze_1043: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1042, 3);  unsqueeze_1042 = None
        mul_1170: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_125);  primals_125 = None
        unsqueeze_1044: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1170, 0);  mul_1170 = None
        unsqueeze_1045: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1044, 2);  unsqueeze_1044 = None
        unsqueeze_1046: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1045, 3);  unsqueeze_1045 = None
        mul_1171: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_1043);  sub_312 = unsqueeze_1043 = None
        sub_314: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_54, mul_1171);  where_54 = mul_1171 = None
        sub_315: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_314, unsqueeze_1040);  sub_314 = unsqueeze_1040 = None
        mul_1172: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_315, unsqueeze_1046);  sub_315 = unsqueeze_1046 = None
        mul_1173: "f32[160]" = torch.ops.aten.mul.Tensor(sum_112, squeeze_124);  sum_112 = squeeze_124 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_1172, cat_5, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1172 = primals_124 = None
        getitem_362: "f32[32, 768, 17, 17]" = convolution_backward_54[0]
        getitem_363: "f32[160, 768, 1, 1]" = convolution_backward_54[1];  convolution_backward_54 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_503: "f32[32, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_502, getitem_362);  add_502 = getitem_362 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_55: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_55, full_default, slice_31);  le_55 = slice_31 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_113: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
        sub_316: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_1049);  convolution_40 = unsqueeze_1049 = None
        mul_1174: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_55, sub_316)
        sum_114: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1174, [0, 2, 3]);  mul_1174 = None
        mul_1175: "f32[192]" = torch.ops.aten.mul.Tensor(sum_113, 0.00010813148788927336)
        unsqueeze_1050: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1175, 0);  mul_1175 = None
        unsqueeze_1051: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, 2);  unsqueeze_1050 = None
        unsqueeze_1052: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1051, 3);  unsqueeze_1051 = None
        mul_1176: "f32[192]" = torch.ops.aten.mul.Tensor(sum_114, 0.00010813148788927336)
        mul_1177: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
        mul_1178: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1176, mul_1177);  mul_1176 = mul_1177 = None
        unsqueeze_1053: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1178, 0);  mul_1178 = None
        unsqueeze_1054: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1053, 2);  unsqueeze_1053 = None
        unsqueeze_1055: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, 3);  unsqueeze_1054 = None
        mul_1179: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_122);  primals_122 = None
        unsqueeze_1056: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1179, 0);  mul_1179 = None
        unsqueeze_1057: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1056, 2);  unsqueeze_1056 = None
        unsqueeze_1058: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1057, 3);  unsqueeze_1057 = None
        mul_1180: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_1055);  sub_316 = unsqueeze_1055 = None
        sub_318: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_55, mul_1180);  where_55 = mul_1180 = None
        sub_319: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_318, unsqueeze_1052);  sub_318 = unsqueeze_1052 = None
        mul_1181: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_1058);  sub_319 = unsqueeze_1058 = None
        mul_1182: "f32[192]" = torch.ops.aten.mul.Tensor(sum_114, squeeze_121);  sum_114 = squeeze_121 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_1181, cat_5, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1181 = cat_5 = primals_121 = None
        getitem_365: "f32[32, 768, 17, 17]" = convolution_backward_55[0]
        getitem_366: "f32[192, 768, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_504: "f32[32, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_503, getitem_365);  add_503 = getitem_365 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:286 in forward, code: return torch.cat(outputs, 1)
        slice_35: "f32[32, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_504, 1, 0, 192)
        slice_36: "f32[32, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_504, 1, 192, 384)
        slice_37: "f32[32, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_504, 1, 384, 576)
        slice_38: "f32[32, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_504, 1, 576, 768);  add_504 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_56: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_56, full_default, slice_38);  le_56 = slice_38 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_115: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
        sub_320: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_1061);  convolution_39 = unsqueeze_1061 = None
        mul_1183: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_56, sub_320)
        sum_116: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1183, [0, 2, 3]);  mul_1183 = None
        mul_1184: "f32[192]" = torch.ops.aten.mul.Tensor(sum_115, 0.00010813148788927336)
        unsqueeze_1062: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1184, 0);  mul_1184 = None
        unsqueeze_1063: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, 2);  unsqueeze_1062 = None
        unsqueeze_1064: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1063, 3);  unsqueeze_1063 = None
        mul_1185: "f32[192]" = torch.ops.aten.mul.Tensor(sum_116, 0.00010813148788927336)
        mul_1186: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
        mul_1187: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1185, mul_1186);  mul_1185 = mul_1186 = None
        unsqueeze_1065: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1187, 0);  mul_1187 = None
        unsqueeze_1066: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1065, 2);  unsqueeze_1065 = None
        unsqueeze_1067: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, 3);  unsqueeze_1066 = None
        mul_1188: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_119);  primals_119 = None
        unsqueeze_1068: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1188, 0);  mul_1188 = None
        unsqueeze_1069: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1068, 2);  unsqueeze_1068 = None
        unsqueeze_1070: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1069, 3);  unsqueeze_1069 = None
        mul_1189: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_1067);  sub_320 = unsqueeze_1067 = None
        sub_322: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_56, mul_1189);  where_56 = mul_1189 = None
        sub_323: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_322, unsqueeze_1064);  sub_322 = unsqueeze_1064 = None
        mul_1190: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_323, unsqueeze_1070);  sub_323 = unsqueeze_1070 = None
        mul_1191: "f32[192]" = torch.ops.aten.mul.Tensor(sum_116, squeeze_118);  sum_116 = squeeze_118 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_1190, avg_pool2d_3, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1190 = avg_pool2d_3 = primals_118 = None
        getitem_368: "f32[32, 768, 17, 17]" = convolution_backward_56[0]
        getitem_369: "f32[192, 768, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:278 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward_6: "f32[32, 768, 17, 17]" = torch.ops.aten.avg_pool2d_backward.default(getitem_368, cat_4, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_368 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_57: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_57, full_default, slice_37);  le_57 = slice_37 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_117: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
        sub_324: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_1073);  convolution_38 = unsqueeze_1073 = None
        mul_1192: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_57, sub_324)
        sum_118: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1192, [0, 2, 3]);  mul_1192 = None
        mul_1193: "f32[192]" = torch.ops.aten.mul.Tensor(sum_117, 0.00010813148788927336)
        unsqueeze_1074: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1193, 0);  mul_1193 = None
        unsqueeze_1075: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, 2);  unsqueeze_1074 = None
        unsqueeze_1076: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1075, 3);  unsqueeze_1075 = None
        mul_1194: "f32[192]" = torch.ops.aten.mul.Tensor(sum_118, 0.00010813148788927336)
        mul_1195: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
        mul_1196: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1194, mul_1195);  mul_1194 = mul_1195 = None
        unsqueeze_1077: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1196, 0);  mul_1196 = None
        unsqueeze_1078: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1077, 2);  unsqueeze_1077 = None
        unsqueeze_1079: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1078, 3);  unsqueeze_1078 = None
        mul_1197: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_116);  primals_116 = None
        unsqueeze_1080: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1197, 0);  mul_1197 = None
        unsqueeze_1081: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1080, 2);  unsqueeze_1080 = None
        unsqueeze_1082: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1081, 3);  unsqueeze_1081 = None
        mul_1198: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_1079);  sub_324 = unsqueeze_1079 = None
        sub_326: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_57, mul_1198);  where_57 = mul_1198 = None
        sub_327: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_326, unsqueeze_1076);  sub_326 = unsqueeze_1076 = None
        mul_1199: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_327, unsqueeze_1082);  sub_327 = unsqueeze_1082 = None
        mul_1200: "f32[192]" = torch.ops.aten.mul.Tensor(sum_118, squeeze_115);  sum_118 = squeeze_115 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_1199, relu_37, primals_115, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1199 = primals_115 = None
        getitem_371: "f32[32, 128, 17, 17]" = convolution_backward_57[0]
        getitem_372: "f32[192, 128, 1, 7]" = convolution_backward_57[1];  convolution_backward_57 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_426: "f32[32, 128, 17, 17]" = torch.ops.aten.alias.default(relu_37);  relu_37 = None
        alias_427: "f32[32, 128, 17, 17]" = torch.ops.aten.alias.default(alias_426);  alias_426 = None
        le_58: "b8[32, 128, 17, 17]" = torch.ops.aten.le.Scalar(alias_427, 0);  alias_427 = None
        where_58: "f32[32, 128, 17, 17]" = torch.ops.aten.where.self(le_58, full_default, getitem_371);  le_58 = getitem_371 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_119: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
        sub_328: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_1085);  convolution_37 = unsqueeze_1085 = None
        mul_1201: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(where_58, sub_328)
        sum_120: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1201, [0, 2, 3]);  mul_1201 = None
        mul_1202: "f32[128]" = torch.ops.aten.mul.Tensor(sum_119, 0.00010813148788927336)
        unsqueeze_1086: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1202, 0);  mul_1202 = None
        unsqueeze_1087: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, 2);  unsqueeze_1086 = None
        unsqueeze_1088: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1087, 3);  unsqueeze_1087 = None
        mul_1203: "f32[128]" = torch.ops.aten.mul.Tensor(sum_120, 0.00010813148788927336)
        mul_1204: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_1205: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1203, mul_1204);  mul_1203 = mul_1204 = None
        unsqueeze_1089: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1205, 0);  mul_1205 = None
        unsqueeze_1090: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1089, 2);  unsqueeze_1089 = None
        unsqueeze_1091: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1090, 3);  unsqueeze_1090 = None
        mul_1206: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_113);  primals_113 = None
        unsqueeze_1092: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1206, 0);  mul_1206 = None
        unsqueeze_1093: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1092, 2);  unsqueeze_1092 = None
        unsqueeze_1094: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1093, 3);  unsqueeze_1093 = None
        mul_1207: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_1091);  sub_328 = unsqueeze_1091 = None
        sub_330: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(where_58, mul_1207);  where_58 = mul_1207 = None
        sub_331: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(sub_330, unsqueeze_1088);  sub_330 = unsqueeze_1088 = None
        mul_1208: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_331, unsqueeze_1094);  sub_331 = unsqueeze_1094 = None
        mul_1209: "f32[128]" = torch.ops.aten.mul.Tensor(sum_120, squeeze_112);  sum_120 = squeeze_112 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_1208, relu_36, primals_112, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1208 = primals_112 = None
        getitem_374: "f32[32, 128, 17, 17]" = convolution_backward_58[0]
        getitem_375: "f32[128, 128, 7, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_430: "f32[32, 128, 17, 17]" = torch.ops.aten.alias.default(relu_36);  relu_36 = None
        alias_431: "f32[32, 128, 17, 17]" = torch.ops.aten.alias.default(alias_430);  alias_430 = None
        le_59: "b8[32, 128, 17, 17]" = torch.ops.aten.le.Scalar(alias_431, 0);  alias_431 = None
        where_59: "f32[32, 128, 17, 17]" = torch.ops.aten.where.self(le_59, full_default, getitem_374);  le_59 = getitem_374 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_121: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
        sub_332: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_1097);  convolution_36 = unsqueeze_1097 = None
        mul_1210: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(where_59, sub_332)
        sum_122: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1210, [0, 2, 3]);  mul_1210 = None
        mul_1211: "f32[128]" = torch.ops.aten.mul.Tensor(sum_121, 0.00010813148788927336)
        unsqueeze_1098: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1211, 0);  mul_1211 = None
        unsqueeze_1099: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, 2);  unsqueeze_1098 = None
        unsqueeze_1100: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1099, 3);  unsqueeze_1099 = None
        mul_1212: "f32[128]" = torch.ops.aten.mul.Tensor(sum_122, 0.00010813148788927336)
        mul_1213: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_1214: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1212, mul_1213);  mul_1212 = mul_1213 = None
        unsqueeze_1101: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1214, 0);  mul_1214 = None
        unsqueeze_1102: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1101, 2);  unsqueeze_1101 = None
        unsqueeze_1103: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1102, 3);  unsqueeze_1102 = None
        mul_1215: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_110);  primals_110 = None
        unsqueeze_1104: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1215, 0);  mul_1215 = None
        unsqueeze_1105: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1104, 2);  unsqueeze_1104 = None
        unsqueeze_1106: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1105, 3);  unsqueeze_1105 = None
        mul_1216: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_332, unsqueeze_1103);  sub_332 = unsqueeze_1103 = None
        sub_334: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(where_59, mul_1216);  where_59 = mul_1216 = None
        sub_335: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(sub_334, unsqueeze_1100);  sub_334 = unsqueeze_1100 = None
        mul_1217: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_335, unsqueeze_1106);  sub_335 = unsqueeze_1106 = None
        mul_1218: "f32[128]" = torch.ops.aten.mul.Tensor(sum_122, squeeze_109);  sum_122 = squeeze_109 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_1217, relu_35, primals_109, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1217 = primals_109 = None
        getitem_377: "f32[32, 128, 17, 17]" = convolution_backward_59[0]
        getitem_378: "f32[128, 128, 1, 7]" = convolution_backward_59[1];  convolution_backward_59 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_434: "f32[32, 128, 17, 17]" = torch.ops.aten.alias.default(relu_35);  relu_35 = None
        alias_435: "f32[32, 128, 17, 17]" = torch.ops.aten.alias.default(alias_434);  alias_434 = None
        le_60: "b8[32, 128, 17, 17]" = torch.ops.aten.le.Scalar(alias_435, 0);  alias_435 = None
        where_60: "f32[32, 128, 17, 17]" = torch.ops.aten.where.self(le_60, full_default, getitem_377);  le_60 = getitem_377 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_123: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
        sub_336: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_1109);  convolution_35 = unsqueeze_1109 = None
        mul_1219: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(where_60, sub_336)
        sum_124: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1219, [0, 2, 3]);  mul_1219 = None
        mul_1220: "f32[128]" = torch.ops.aten.mul.Tensor(sum_123, 0.00010813148788927336)
        unsqueeze_1110: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1220, 0);  mul_1220 = None
        unsqueeze_1111: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, 2);  unsqueeze_1110 = None
        unsqueeze_1112: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1111, 3);  unsqueeze_1111 = None
        mul_1221: "f32[128]" = torch.ops.aten.mul.Tensor(sum_124, 0.00010813148788927336)
        mul_1222: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_1223: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1221, mul_1222);  mul_1221 = mul_1222 = None
        unsqueeze_1113: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1223, 0);  mul_1223 = None
        unsqueeze_1114: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1113, 2);  unsqueeze_1113 = None
        unsqueeze_1115: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1114, 3);  unsqueeze_1114 = None
        mul_1224: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_107);  primals_107 = None
        unsqueeze_1116: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1224, 0);  mul_1224 = None
        unsqueeze_1117: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1116, 2);  unsqueeze_1116 = None
        unsqueeze_1118: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1117, 3);  unsqueeze_1117 = None
        mul_1225: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_336, unsqueeze_1115);  sub_336 = unsqueeze_1115 = None
        sub_338: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(where_60, mul_1225);  where_60 = mul_1225 = None
        sub_339: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(sub_338, unsqueeze_1112);  sub_338 = unsqueeze_1112 = None
        mul_1226: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_339, unsqueeze_1118);  sub_339 = unsqueeze_1118 = None
        mul_1227: "f32[128]" = torch.ops.aten.mul.Tensor(sum_124, squeeze_106);  sum_124 = squeeze_106 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1226, relu_34, primals_106, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1226 = primals_106 = None
        getitem_380: "f32[32, 128, 17, 17]" = convolution_backward_60[0]
        getitem_381: "f32[128, 128, 7, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_438: "f32[32, 128, 17, 17]" = torch.ops.aten.alias.default(relu_34);  relu_34 = None
        alias_439: "f32[32, 128, 17, 17]" = torch.ops.aten.alias.default(alias_438);  alias_438 = None
        le_61: "b8[32, 128, 17, 17]" = torch.ops.aten.le.Scalar(alias_439, 0);  alias_439 = None
        where_61: "f32[32, 128, 17, 17]" = torch.ops.aten.where.self(le_61, full_default, getitem_380);  le_61 = getitem_380 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_125: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
        sub_340: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_1121);  convolution_34 = unsqueeze_1121 = None
        mul_1228: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(where_61, sub_340)
        sum_126: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1228, [0, 2, 3]);  mul_1228 = None
        mul_1229: "f32[128]" = torch.ops.aten.mul.Tensor(sum_125, 0.00010813148788927336)
        unsqueeze_1122: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1229, 0);  mul_1229 = None
        unsqueeze_1123: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, 2);  unsqueeze_1122 = None
        unsqueeze_1124: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1123, 3);  unsqueeze_1123 = None
        mul_1230: "f32[128]" = torch.ops.aten.mul.Tensor(sum_126, 0.00010813148788927336)
        mul_1231: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_1232: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1230, mul_1231);  mul_1230 = mul_1231 = None
        unsqueeze_1125: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1232, 0);  mul_1232 = None
        unsqueeze_1126: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1125, 2);  unsqueeze_1125 = None
        unsqueeze_1127: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1126, 3);  unsqueeze_1126 = None
        mul_1233: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_104);  primals_104 = None
        unsqueeze_1128: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1233, 0);  mul_1233 = None
        unsqueeze_1129: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1128, 2);  unsqueeze_1128 = None
        unsqueeze_1130: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1129, 3);  unsqueeze_1129 = None
        mul_1234: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_340, unsqueeze_1127);  sub_340 = unsqueeze_1127 = None
        sub_342: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(where_61, mul_1234);  where_61 = mul_1234 = None
        sub_343: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(sub_342, unsqueeze_1124);  sub_342 = unsqueeze_1124 = None
        mul_1235: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_343, unsqueeze_1130);  sub_343 = unsqueeze_1130 = None
        mul_1236: "f32[128]" = torch.ops.aten.mul.Tensor(sum_126, squeeze_103);  sum_126 = squeeze_103 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1235, cat_4, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1235 = primals_103 = None
        getitem_383: "f32[32, 768, 17, 17]" = convolution_backward_61[0]
        getitem_384: "f32[128, 768, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_505: "f32[32, 768, 17, 17]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_6, getitem_383);  avg_pool2d_backward_6 = getitem_383 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_62: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_62, full_default, slice_36);  le_62 = slice_36 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_127: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
        sub_344: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_1133);  convolution_33 = unsqueeze_1133 = None
        mul_1237: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_62, sub_344)
        sum_128: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1237, [0, 2, 3]);  mul_1237 = None
        mul_1238: "f32[192]" = torch.ops.aten.mul.Tensor(sum_127, 0.00010813148788927336)
        unsqueeze_1134: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1238, 0);  mul_1238 = None
        unsqueeze_1135: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, 2);  unsqueeze_1134 = None
        unsqueeze_1136: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1135, 3);  unsqueeze_1135 = None
        mul_1239: "f32[192]" = torch.ops.aten.mul.Tensor(sum_128, 0.00010813148788927336)
        mul_1240: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_1241: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1239, mul_1240);  mul_1239 = mul_1240 = None
        unsqueeze_1137: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1241, 0);  mul_1241 = None
        unsqueeze_1138: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1137, 2);  unsqueeze_1137 = None
        unsqueeze_1139: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1138, 3);  unsqueeze_1138 = None
        mul_1242: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_101);  primals_101 = None
        unsqueeze_1140: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1242, 0);  mul_1242 = None
        unsqueeze_1141: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1140, 2);  unsqueeze_1140 = None
        unsqueeze_1142: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1141, 3);  unsqueeze_1141 = None
        mul_1243: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_344, unsqueeze_1139);  sub_344 = unsqueeze_1139 = None
        sub_346: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_62, mul_1243);  where_62 = mul_1243 = None
        sub_347: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_346, unsqueeze_1136);  sub_346 = unsqueeze_1136 = None
        mul_1244: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_347, unsqueeze_1142);  sub_347 = unsqueeze_1142 = None
        mul_1245: "f32[192]" = torch.ops.aten.mul.Tensor(sum_128, squeeze_100);  sum_128 = squeeze_100 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1244, relu_32, primals_100, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1244 = primals_100 = None
        getitem_386: "f32[32, 128, 17, 17]" = convolution_backward_62[0]
        getitem_387: "f32[192, 128, 7, 1]" = convolution_backward_62[1];  convolution_backward_62 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_446: "f32[32, 128, 17, 17]" = torch.ops.aten.alias.default(relu_32);  relu_32 = None
        alias_447: "f32[32, 128, 17, 17]" = torch.ops.aten.alias.default(alias_446);  alias_446 = None
        le_63: "b8[32, 128, 17, 17]" = torch.ops.aten.le.Scalar(alias_447, 0);  alias_447 = None
        where_63: "f32[32, 128, 17, 17]" = torch.ops.aten.where.self(le_63, full_default, getitem_386);  le_63 = getitem_386 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_129: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_63, [0, 2, 3])
        sub_348: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_1145);  convolution_32 = unsqueeze_1145 = None
        mul_1246: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(where_63, sub_348)
        sum_130: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1246, [0, 2, 3]);  mul_1246 = None
        mul_1247: "f32[128]" = torch.ops.aten.mul.Tensor(sum_129, 0.00010813148788927336)
        unsqueeze_1146: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1247, 0);  mul_1247 = None
        unsqueeze_1147: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, 2);  unsqueeze_1146 = None
        unsqueeze_1148: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1147, 3);  unsqueeze_1147 = None
        mul_1248: "f32[128]" = torch.ops.aten.mul.Tensor(sum_130, 0.00010813148788927336)
        mul_1249: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_1250: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1248, mul_1249);  mul_1248 = mul_1249 = None
        unsqueeze_1149: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1250, 0);  mul_1250 = None
        unsqueeze_1150: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1149, 2);  unsqueeze_1149 = None
        unsqueeze_1151: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1150, 3);  unsqueeze_1150 = None
        mul_1251: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_98);  primals_98 = None
        unsqueeze_1152: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1251, 0);  mul_1251 = None
        unsqueeze_1153: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1152, 2);  unsqueeze_1152 = None
        unsqueeze_1154: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1153, 3);  unsqueeze_1153 = None
        mul_1252: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_348, unsqueeze_1151);  sub_348 = unsqueeze_1151 = None
        sub_350: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(where_63, mul_1252);  where_63 = mul_1252 = None
        sub_351: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(sub_350, unsqueeze_1148);  sub_350 = unsqueeze_1148 = None
        mul_1253: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_351, unsqueeze_1154);  sub_351 = unsqueeze_1154 = None
        mul_1254: "f32[128]" = torch.ops.aten.mul.Tensor(sum_130, squeeze_97);  sum_130 = squeeze_97 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1253, relu_31, primals_97, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1253 = primals_97 = None
        getitem_389: "f32[32, 128, 17, 17]" = convolution_backward_63[0]
        getitem_390: "f32[128, 128, 1, 7]" = convolution_backward_63[1];  convolution_backward_63 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_450: "f32[32, 128, 17, 17]" = torch.ops.aten.alias.default(relu_31);  relu_31 = None
        alias_451: "f32[32, 128, 17, 17]" = torch.ops.aten.alias.default(alias_450);  alias_450 = None
        le_64: "b8[32, 128, 17, 17]" = torch.ops.aten.le.Scalar(alias_451, 0);  alias_451 = None
        where_64: "f32[32, 128, 17, 17]" = torch.ops.aten.where.self(le_64, full_default, getitem_389);  le_64 = getitem_389 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_131: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
        sub_352: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_1157);  convolution_31 = unsqueeze_1157 = None
        mul_1255: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(where_64, sub_352)
        sum_132: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1255, [0, 2, 3]);  mul_1255 = None
        mul_1256: "f32[128]" = torch.ops.aten.mul.Tensor(sum_131, 0.00010813148788927336)
        unsqueeze_1158: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1256, 0);  mul_1256 = None
        unsqueeze_1159: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, 2);  unsqueeze_1158 = None
        unsqueeze_1160: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1159, 3);  unsqueeze_1159 = None
        mul_1257: "f32[128]" = torch.ops.aten.mul.Tensor(sum_132, 0.00010813148788927336)
        mul_1258: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_1259: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1257, mul_1258);  mul_1257 = mul_1258 = None
        unsqueeze_1161: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1259, 0);  mul_1259 = None
        unsqueeze_1162: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1161, 2);  unsqueeze_1161 = None
        unsqueeze_1163: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1162, 3);  unsqueeze_1162 = None
        mul_1260: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_95);  primals_95 = None
        unsqueeze_1164: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1260, 0);  mul_1260 = None
        unsqueeze_1165: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1164, 2);  unsqueeze_1164 = None
        unsqueeze_1166: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1165, 3);  unsqueeze_1165 = None
        mul_1261: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_352, unsqueeze_1163);  sub_352 = unsqueeze_1163 = None
        sub_354: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(where_64, mul_1261);  where_64 = mul_1261 = None
        sub_355: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(sub_354, unsqueeze_1160);  sub_354 = unsqueeze_1160 = None
        mul_1262: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_355, unsqueeze_1166);  sub_355 = unsqueeze_1166 = None
        mul_1263: "f32[128]" = torch.ops.aten.mul.Tensor(sum_132, squeeze_94);  sum_132 = squeeze_94 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1262, cat_4, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1262 = primals_94 = None
        getitem_392: "f32[32, 768, 17, 17]" = convolution_backward_64[0]
        getitem_393: "f32[128, 768, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_506: "f32[32, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_505, getitem_392);  add_505 = getitem_392 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_65: "f32[32, 192, 17, 17]" = torch.ops.aten.where.self(le_65, full_default, slice_35);  le_65 = slice_35 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_133: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_65, [0, 2, 3])
        sub_356: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_1169);  convolution_30 = unsqueeze_1169 = None
        mul_1264: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_65, sub_356)
        sum_134: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1264, [0, 2, 3]);  mul_1264 = None
        mul_1265: "f32[192]" = torch.ops.aten.mul.Tensor(sum_133, 0.00010813148788927336)
        unsqueeze_1170: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1265, 0);  mul_1265 = None
        unsqueeze_1171: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, 2);  unsqueeze_1170 = None
        unsqueeze_1172: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1171, 3);  unsqueeze_1171 = None
        mul_1266: "f32[192]" = torch.ops.aten.mul.Tensor(sum_134, 0.00010813148788927336)
        mul_1267: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_1268: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1266, mul_1267);  mul_1266 = mul_1267 = None
        unsqueeze_1173: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1268, 0);  mul_1268 = None
        unsqueeze_1174: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1173, 2);  unsqueeze_1173 = None
        unsqueeze_1175: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1174, 3);  unsqueeze_1174 = None
        mul_1269: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_92);  primals_92 = None
        unsqueeze_1176: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1269, 0);  mul_1269 = None
        unsqueeze_1177: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1176, 2);  unsqueeze_1176 = None
        unsqueeze_1178: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1177, 3);  unsqueeze_1177 = None
        mul_1270: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_356, unsqueeze_1175);  sub_356 = unsqueeze_1175 = None
        sub_358: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_65, mul_1270);  where_65 = mul_1270 = None
        sub_359: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_358, unsqueeze_1172);  sub_358 = unsqueeze_1172 = None
        mul_1271: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_359, unsqueeze_1178);  sub_359 = unsqueeze_1178 = None
        mul_1272: "f32[192]" = torch.ops.aten.mul.Tensor(sum_134, squeeze_91);  sum_134 = squeeze_91 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_1271, cat_4, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1271 = cat_4 = primals_91 = None
        getitem_395: "f32[32, 768, 17, 17]" = convolution_backward_65[0]
        getitem_396: "f32[192, 768, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_507: "f32[32, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_506, getitem_395);  add_506 = getitem_395 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:240 in forward, code: return torch.cat(outputs, 1)
        slice_39: "f32[32, 384, 17, 17]" = torch.ops.aten.slice.Tensor(add_507, 1, 0, 384)
        slice_40: "f32[32, 96, 17, 17]" = torch.ops.aten.slice.Tensor(add_507, 1, 384, 480)
        slice_41: "f32[32, 288, 17, 17]" = torch.ops.aten.slice.Tensor(add_507, 1, 480, 768);  add_507 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:233 in _forward, code: branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        max_pool2d_with_indices_backward_1: "f32[32, 288, 35, 35]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_41, cat_3, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_65);  slice_41 = getitem_65 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_66: "f32[32, 96, 17, 17]" = torch.ops.aten.where.self(le_66, full_default, slice_40);  le_66 = slice_40 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_135: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
        sub_360: "f32[32, 96, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_1181);  convolution_29 = unsqueeze_1181 = None
        mul_1273: "f32[32, 96, 17, 17]" = torch.ops.aten.mul.Tensor(where_66, sub_360)
        sum_136: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1273, [0, 2, 3]);  mul_1273 = None
        mul_1274: "f32[96]" = torch.ops.aten.mul.Tensor(sum_135, 0.00010813148788927336)
        unsqueeze_1182: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1274, 0);  mul_1274 = None
        unsqueeze_1183: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, 2);  unsqueeze_1182 = None
        unsqueeze_1184: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1183, 3);  unsqueeze_1183 = None
        mul_1275: "f32[96]" = torch.ops.aten.mul.Tensor(sum_136, 0.00010813148788927336)
        mul_1276: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_1277: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1275, mul_1276);  mul_1275 = mul_1276 = None
        unsqueeze_1185: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1277, 0);  mul_1277 = None
        unsqueeze_1186: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1185, 2);  unsqueeze_1185 = None
        unsqueeze_1187: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1186, 3);  unsqueeze_1186 = None
        mul_1278: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_89);  primals_89 = None
        unsqueeze_1188: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1278, 0);  mul_1278 = None
        unsqueeze_1189: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1188, 2);  unsqueeze_1188 = None
        unsqueeze_1190: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1189, 3);  unsqueeze_1189 = None
        mul_1279: "f32[32, 96, 17, 17]" = torch.ops.aten.mul.Tensor(sub_360, unsqueeze_1187);  sub_360 = unsqueeze_1187 = None
        sub_362: "f32[32, 96, 17, 17]" = torch.ops.aten.sub.Tensor(where_66, mul_1279);  where_66 = mul_1279 = None
        sub_363: "f32[32, 96, 17, 17]" = torch.ops.aten.sub.Tensor(sub_362, unsqueeze_1184);  sub_362 = unsqueeze_1184 = None
        mul_1280: "f32[32, 96, 17, 17]" = torch.ops.aten.mul.Tensor(sub_363, unsqueeze_1190);  sub_363 = unsqueeze_1190 = None
        mul_1281: "f32[96]" = torch.ops.aten.mul.Tensor(sum_136, squeeze_88);  sum_136 = squeeze_88 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1280, relu_28, primals_88, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1280 = primals_88 = None
        getitem_398: "f32[32, 96, 35, 35]" = convolution_backward_66[0]
        getitem_399: "f32[96, 96, 3, 3]" = convolution_backward_66[1];  convolution_backward_66 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_462: "f32[32, 96, 35, 35]" = torch.ops.aten.alias.default(relu_28);  relu_28 = None
        alias_463: "f32[32, 96, 35, 35]" = torch.ops.aten.alias.default(alias_462);  alias_462 = None
        le_67: "b8[32, 96, 35, 35]" = torch.ops.aten.le.Scalar(alias_463, 0);  alias_463 = None
        where_67: "f32[32, 96, 35, 35]" = torch.ops.aten.where.self(le_67, full_default, getitem_398);  le_67 = getitem_398 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_137: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_67, [0, 2, 3])
        sub_364: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_1193);  convolution_28 = unsqueeze_1193 = None
        mul_1282: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(where_67, sub_364)
        sum_138: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1282, [0, 2, 3]);  mul_1282 = None
        mul_1283: "f32[96]" = torch.ops.aten.mul.Tensor(sum_137, 2.5510204081632654e-05)
        unsqueeze_1194: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1283, 0);  mul_1283 = None
        unsqueeze_1195: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, 2);  unsqueeze_1194 = None
        unsqueeze_1196: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1195, 3);  unsqueeze_1195 = None
        mul_1284: "f32[96]" = torch.ops.aten.mul.Tensor(sum_138, 2.5510204081632654e-05)
        mul_1285: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_1286: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1284, mul_1285);  mul_1284 = mul_1285 = None
        unsqueeze_1197: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1286, 0);  mul_1286 = None
        unsqueeze_1198: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1197, 2);  unsqueeze_1197 = None
        unsqueeze_1199: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1198, 3);  unsqueeze_1198 = None
        mul_1287: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_86);  primals_86 = None
        unsqueeze_1200: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1287, 0);  mul_1287 = None
        unsqueeze_1201: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1200, 2);  unsqueeze_1200 = None
        unsqueeze_1202: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1201, 3);  unsqueeze_1201 = None
        mul_1288: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_364, unsqueeze_1199);  sub_364 = unsqueeze_1199 = None
        sub_366: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(where_67, mul_1288);  where_67 = mul_1288 = None
        sub_367: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(sub_366, unsqueeze_1196);  sub_366 = unsqueeze_1196 = None
        mul_1289: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_367, unsqueeze_1202);  sub_367 = unsqueeze_1202 = None
        mul_1290: "f32[96]" = torch.ops.aten.mul.Tensor(sum_138, squeeze_85);  sum_138 = squeeze_85 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_1289, relu_27, primals_85, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1289 = primals_85 = None
        getitem_401: "f32[32, 64, 35, 35]" = convolution_backward_67[0]
        getitem_402: "f32[96, 64, 3, 3]" = convolution_backward_67[1];  convolution_backward_67 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_466: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
        alias_467: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(alias_466);  alias_466 = None
        le_68: "b8[32, 64, 35, 35]" = torch.ops.aten.le.Scalar(alias_467, 0);  alias_467 = None
        where_68: "f32[32, 64, 35, 35]" = torch.ops.aten.where.self(le_68, full_default, getitem_401);  le_68 = getitem_401 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_139: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
        sub_368: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_1205);  convolution_27 = unsqueeze_1205 = None
        mul_1291: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_68, sub_368)
        sum_140: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1291, [0, 2, 3]);  mul_1291 = None
        mul_1292: "f32[64]" = torch.ops.aten.mul.Tensor(sum_139, 2.5510204081632654e-05)
        unsqueeze_1206: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1292, 0);  mul_1292 = None
        unsqueeze_1207: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, 2);  unsqueeze_1206 = None
        unsqueeze_1208: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1207, 3);  unsqueeze_1207 = None
        mul_1293: "f32[64]" = torch.ops.aten.mul.Tensor(sum_140, 2.5510204081632654e-05)
        mul_1294: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_1295: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1293, mul_1294);  mul_1293 = mul_1294 = None
        unsqueeze_1209: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1295, 0);  mul_1295 = None
        unsqueeze_1210: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1209, 2);  unsqueeze_1209 = None
        unsqueeze_1211: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1210, 3);  unsqueeze_1210 = None
        mul_1296: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_83);  primals_83 = None
        unsqueeze_1212: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1296, 0);  mul_1296 = None
        unsqueeze_1213: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1212, 2);  unsqueeze_1212 = None
        unsqueeze_1214: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1213, 3);  unsqueeze_1213 = None
        mul_1297: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_368, unsqueeze_1211);  sub_368 = unsqueeze_1211 = None
        sub_370: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_68, mul_1297);  where_68 = mul_1297 = None
        sub_371: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_370, unsqueeze_1208);  sub_370 = unsqueeze_1208 = None
        mul_1298: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_371, unsqueeze_1214);  sub_371 = unsqueeze_1214 = None
        mul_1299: "f32[64]" = torch.ops.aten.mul.Tensor(sum_140, squeeze_82);  sum_140 = squeeze_82 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_1298, cat_3, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1298 = primals_82 = None
        getitem_404: "f32[32, 288, 35, 35]" = convolution_backward_68[0]
        getitem_405: "f32[64, 288, 1, 1]" = convolution_backward_68[1];  convolution_backward_68 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_508: "f32[32, 288, 35, 35]" = torch.ops.aten.add.Tensor(max_pool2d_with_indices_backward_1, getitem_404);  max_pool2d_with_indices_backward_1 = getitem_404 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_69: "f32[32, 384, 17, 17]" = torch.ops.aten.where.self(le_69, full_default, slice_39);  le_69 = slice_39 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_141: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_69, [0, 2, 3])
        sub_372: "f32[32, 384, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_1217);  convolution_26 = unsqueeze_1217 = None
        mul_1300: "f32[32, 384, 17, 17]" = torch.ops.aten.mul.Tensor(where_69, sub_372)
        sum_142: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_1300, [0, 2, 3]);  mul_1300 = None
        mul_1301: "f32[384]" = torch.ops.aten.mul.Tensor(sum_141, 0.00010813148788927336)
        unsqueeze_1218: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_1301, 0);  mul_1301 = None
        unsqueeze_1219: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, 2);  unsqueeze_1218 = None
        unsqueeze_1220: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1219, 3);  unsqueeze_1219 = None
        mul_1302: "f32[384]" = torch.ops.aten.mul.Tensor(sum_142, 0.00010813148788927336)
        mul_1303: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_1304: "f32[384]" = torch.ops.aten.mul.Tensor(mul_1302, mul_1303);  mul_1302 = mul_1303 = None
        unsqueeze_1221: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_1304, 0);  mul_1304 = None
        unsqueeze_1222: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1221, 2);  unsqueeze_1221 = None
        unsqueeze_1223: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1222, 3);  unsqueeze_1222 = None
        mul_1305: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_80);  primals_80 = None
        unsqueeze_1224: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_1305, 0);  mul_1305 = None
        unsqueeze_1225: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1224, 2);  unsqueeze_1224 = None
        unsqueeze_1226: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1225, 3);  unsqueeze_1225 = None
        mul_1306: "f32[32, 384, 17, 17]" = torch.ops.aten.mul.Tensor(sub_372, unsqueeze_1223);  sub_372 = unsqueeze_1223 = None
        sub_374: "f32[32, 384, 17, 17]" = torch.ops.aten.sub.Tensor(where_69, mul_1306);  where_69 = mul_1306 = None
        sub_375: "f32[32, 384, 17, 17]" = torch.ops.aten.sub.Tensor(sub_374, unsqueeze_1220);  sub_374 = unsqueeze_1220 = None
        mul_1307: "f32[32, 384, 17, 17]" = torch.ops.aten.mul.Tensor(sub_375, unsqueeze_1226);  sub_375 = unsqueeze_1226 = None
        mul_1308: "f32[384]" = torch.ops.aten.mul.Tensor(sum_142, squeeze_79);  sum_142 = squeeze_79 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_1307, cat_3, primals_79, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1307 = cat_3 = primals_79 = None
        getitem_407: "f32[32, 288, 35, 35]" = convolution_backward_69[0]
        getitem_408: "f32[384, 288, 3, 3]" = convolution_backward_69[1];  convolution_backward_69 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_509: "f32[32, 288, 35, 35]" = torch.ops.aten.add.Tensor(add_508, getitem_407);  add_508 = getitem_407 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:212 in forward, code: return torch.cat(outputs, 1)
        slice_42: "f32[32, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_509, 1, 0, 64)
        slice_43: "f32[32, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_509, 1, 64, 128)
        slice_44: "f32[32, 96, 35, 35]" = torch.ops.aten.slice.Tensor(add_509, 1, 128, 224)
        slice_45: "f32[32, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_509, 1, 224, 288);  add_509 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_70: "f32[32, 64, 35, 35]" = torch.ops.aten.where.self(le_70, full_default, slice_45);  le_70 = slice_45 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_143: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_70, [0, 2, 3])
        sub_376: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_1229);  convolution_25 = unsqueeze_1229 = None
        mul_1309: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_70, sub_376)
        sum_144: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1309, [0, 2, 3]);  mul_1309 = None
        mul_1310: "f32[64]" = torch.ops.aten.mul.Tensor(sum_143, 2.5510204081632654e-05)
        unsqueeze_1230: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1310, 0);  mul_1310 = None
        unsqueeze_1231: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, 2);  unsqueeze_1230 = None
        unsqueeze_1232: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1231, 3);  unsqueeze_1231 = None
        mul_1311: "f32[64]" = torch.ops.aten.mul.Tensor(sum_144, 2.5510204081632654e-05)
        mul_1312: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_1313: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1311, mul_1312);  mul_1311 = mul_1312 = None
        unsqueeze_1233: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1313, 0);  mul_1313 = None
        unsqueeze_1234: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1233, 2);  unsqueeze_1233 = None
        unsqueeze_1235: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1234, 3);  unsqueeze_1234 = None
        mul_1314: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_77);  primals_77 = None
        unsqueeze_1236: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1314, 0);  mul_1314 = None
        unsqueeze_1237: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1236, 2);  unsqueeze_1236 = None
        unsqueeze_1238: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1237, 3);  unsqueeze_1237 = None
        mul_1315: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_376, unsqueeze_1235);  sub_376 = unsqueeze_1235 = None
        sub_378: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_70, mul_1315);  where_70 = mul_1315 = None
        sub_379: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_378, unsqueeze_1232);  sub_378 = unsqueeze_1232 = None
        mul_1316: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_379, unsqueeze_1238);  sub_379 = unsqueeze_1238 = None
        mul_1317: "f32[64]" = torch.ops.aten.mul.Tensor(sum_144, squeeze_76);  sum_144 = squeeze_76 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1316, avg_pool2d_2, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1316 = avg_pool2d_2 = primals_76 = None
        getitem_410: "f32[32, 288, 35, 35]" = convolution_backward_70[0]
        getitem_411: "f32[64, 288, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:204 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward_7: "f32[32, 288, 35, 35]" = torch.ops.aten.avg_pool2d_backward.default(getitem_410, cat_2, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_410 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_71: "f32[32, 96, 35, 35]" = torch.ops.aten.where.self(le_71, full_default, slice_44);  le_71 = slice_44 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_145: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_71, [0, 2, 3])
        sub_380: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_1241);  convolution_24 = unsqueeze_1241 = None
        mul_1318: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(where_71, sub_380)
        sum_146: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1318, [0, 2, 3]);  mul_1318 = None
        mul_1319: "f32[96]" = torch.ops.aten.mul.Tensor(sum_145, 2.5510204081632654e-05)
        unsqueeze_1242: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1319, 0);  mul_1319 = None
        unsqueeze_1243: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1242, 2);  unsqueeze_1242 = None
        unsqueeze_1244: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1243, 3);  unsqueeze_1243 = None
        mul_1320: "f32[96]" = torch.ops.aten.mul.Tensor(sum_146, 2.5510204081632654e-05)
        mul_1321: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_1322: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1320, mul_1321);  mul_1320 = mul_1321 = None
        unsqueeze_1245: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1322, 0);  mul_1322 = None
        unsqueeze_1246: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1245, 2);  unsqueeze_1245 = None
        unsqueeze_1247: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1246, 3);  unsqueeze_1246 = None
        mul_1323: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_74);  primals_74 = None
        unsqueeze_1248: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1323, 0);  mul_1323 = None
        unsqueeze_1249: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1248, 2);  unsqueeze_1248 = None
        unsqueeze_1250: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1249, 3);  unsqueeze_1249 = None
        mul_1324: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_380, unsqueeze_1247);  sub_380 = unsqueeze_1247 = None
        sub_382: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(where_71, mul_1324);  where_71 = mul_1324 = None
        sub_383: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(sub_382, unsqueeze_1244);  sub_382 = unsqueeze_1244 = None
        mul_1325: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_383, unsqueeze_1250);  sub_383 = unsqueeze_1250 = None
        mul_1326: "f32[96]" = torch.ops.aten.mul.Tensor(sum_146, squeeze_73);  sum_146 = squeeze_73 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_1325, relu_23, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1325 = primals_73 = None
        getitem_413: "f32[32, 96, 35, 35]" = convolution_backward_71[0]
        getitem_414: "f32[96, 96, 3, 3]" = convolution_backward_71[1];  convolution_backward_71 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_482: "f32[32, 96, 35, 35]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
        alias_483: "f32[32, 96, 35, 35]" = torch.ops.aten.alias.default(alias_482);  alias_482 = None
        le_72: "b8[32, 96, 35, 35]" = torch.ops.aten.le.Scalar(alias_483, 0);  alias_483 = None
        where_72: "f32[32, 96, 35, 35]" = torch.ops.aten.where.self(le_72, full_default, getitem_413);  le_72 = getitem_413 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_147: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
        sub_384: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_1253);  convolution_23 = unsqueeze_1253 = None
        mul_1327: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(where_72, sub_384)
        sum_148: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1327, [0, 2, 3]);  mul_1327 = None
        mul_1328: "f32[96]" = torch.ops.aten.mul.Tensor(sum_147, 2.5510204081632654e-05)
        unsqueeze_1254: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1328, 0);  mul_1328 = None
        unsqueeze_1255: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, 2);  unsqueeze_1254 = None
        unsqueeze_1256: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1255, 3);  unsqueeze_1255 = None
        mul_1329: "f32[96]" = torch.ops.aten.mul.Tensor(sum_148, 2.5510204081632654e-05)
        mul_1330: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_1331: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1329, mul_1330);  mul_1329 = mul_1330 = None
        unsqueeze_1257: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1331, 0);  mul_1331 = None
        unsqueeze_1258: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1257, 2);  unsqueeze_1257 = None
        unsqueeze_1259: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1258, 3);  unsqueeze_1258 = None
        mul_1332: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_71);  primals_71 = None
        unsqueeze_1260: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1332, 0);  mul_1332 = None
        unsqueeze_1261: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1260, 2);  unsqueeze_1260 = None
        unsqueeze_1262: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1261, 3);  unsqueeze_1261 = None
        mul_1333: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_384, unsqueeze_1259);  sub_384 = unsqueeze_1259 = None
        sub_386: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(where_72, mul_1333);  where_72 = mul_1333 = None
        sub_387: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(sub_386, unsqueeze_1256);  sub_386 = unsqueeze_1256 = None
        mul_1334: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_387, unsqueeze_1262);  sub_387 = unsqueeze_1262 = None
        mul_1335: "f32[96]" = torch.ops.aten.mul.Tensor(sum_148, squeeze_70);  sum_148 = squeeze_70 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_1334, relu_22, primals_70, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1334 = primals_70 = None
        getitem_416: "f32[32, 64, 35, 35]" = convolution_backward_72[0]
        getitem_417: "f32[96, 64, 3, 3]" = convolution_backward_72[1];  convolution_backward_72 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_486: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
        alias_487: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(alias_486);  alias_486 = None
        le_73: "b8[32, 64, 35, 35]" = torch.ops.aten.le.Scalar(alias_487, 0);  alias_487 = None
        where_73: "f32[32, 64, 35, 35]" = torch.ops.aten.where.self(le_73, full_default, getitem_416);  le_73 = getitem_416 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_149: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_73, [0, 2, 3])
        sub_388: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_1265);  convolution_22 = unsqueeze_1265 = None
        mul_1336: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_73, sub_388)
        sum_150: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1336, [0, 2, 3]);  mul_1336 = None
        mul_1337: "f32[64]" = torch.ops.aten.mul.Tensor(sum_149, 2.5510204081632654e-05)
        unsqueeze_1266: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1337, 0);  mul_1337 = None
        unsqueeze_1267: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, 2);  unsqueeze_1266 = None
        unsqueeze_1268: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1267, 3);  unsqueeze_1267 = None
        mul_1338: "f32[64]" = torch.ops.aten.mul.Tensor(sum_150, 2.5510204081632654e-05)
        mul_1339: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_1340: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1338, mul_1339);  mul_1338 = mul_1339 = None
        unsqueeze_1269: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1340, 0);  mul_1340 = None
        unsqueeze_1270: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1269, 2);  unsqueeze_1269 = None
        unsqueeze_1271: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1270, 3);  unsqueeze_1270 = None
        mul_1341: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_68);  primals_68 = None
        unsqueeze_1272: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1341, 0);  mul_1341 = None
        unsqueeze_1273: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1272, 2);  unsqueeze_1272 = None
        unsqueeze_1274: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1273, 3);  unsqueeze_1273 = None
        mul_1342: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_388, unsqueeze_1271);  sub_388 = unsqueeze_1271 = None
        sub_390: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_73, mul_1342);  where_73 = mul_1342 = None
        sub_391: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_390, unsqueeze_1268);  sub_390 = unsqueeze_1268 = None
        mul_1343: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_391, unsqueeze_1274);  sub_391 = unsqueeze_1274 = None
        mul_1344: "f32[64]" = torch.ops.aten.mul.Tensor(sum_150, squeeze_67);  sum_150 = squeeze_67 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_1343, cat_2, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1343 = primals_67 = None
        getitem_419: "f32[32, 288, 35, 35]" = convolution_backward_73[0]
        getitem_420: "f32[64, 288, 1, 1]" = convolution_backward_73[1];  convolution_backward_73 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_510: "f32[32, 288, 35, 35]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_7, getitem_419);  avg_pool2d_backward_7 = getitem_419 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_74: "f32[32, 64, 35, 35]" = torch.ops.aten.where.self(le_74, full_default, slice_43);  le_74 = slice_43 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_151: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_74, [0, 2, 3])
        sub_392: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_1277);  convolution_21 = unsqueeze_1277 = None
        mul_1345: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_74, sub_392)
        sum_152: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1345, [0, 2, 3]);  mul_1345 = None
        mul_1346: "f32[64]" = torch.ops.aten.mul.Tensor(sum_151, 2.5510204081632654e-05)
        unsqueeze_1278: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1346, 0);  mul_1346 = None
        unsqueeze_1279: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, 2);  unsqueeze_1278 = None
        unsqueeze_1280: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1279, 3);  unsqueeze_1279 = None
        mul_1347: "f32[64]" = torch.ops.aten.mul.Tensor(sum_152, 2.5510204081632654e-05)
        mul_1348: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_1349: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1347, mul_1348);  mul_1347 = mul_1348 = None
        unsqueeze_1281: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1349, 0);  mul_1349 = None
        unsqueeze_1282: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1281, 2);  unsqueeze_1281 = None
        unsqueeze_1283: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1282, 3);  unsqueeze_1282 = None
        mul_1350: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_65);  primals_65 = None
        unsqueeze_1284: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1350, 0);  mul_1350 = None
        unsqueeze_1285: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1284, 2);  unsqueeze_1284 = None
        unsqueeze_1286: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1285, 3);  unsqueeze_1285 = None
        mul_1351: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_392, unsqueeze_1283);  sub_392 = unsqueeze_1283 = None
        sub_394: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_74, mul_1351);  where_74 = mul_1351 = None
        sub_395: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_394, unsqueeze_1280);  sub_394 = unsqueeze_1280 = None
        mul_1352: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_395, unsqueeze_1286);  sub_395 = unsqueeze_1286 = None
        mul_1353: "f32[64]" = torch.ops.aten.mul.Tensor(sum_152, squeeze_64);  sum_152 = squeeze_64 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1352, relu_20, primals_64, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1352 = primals_64 = None
        getitem_422: "f32[32, 48, 35, 35]" = convolution_backward_74[0]
        getitem_423: "f32[64, 48, 5, 5]" = convolution_backward_74[1];  convolution_backward_74 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_494: "f32[32, 48, 35, 35]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
        alias_495: "f32[32, 48, 35, 35]" = torch.ops.aten.alias.default(alias_494);  alias_494 = None
        le_75: "b8[32, 48, 35, 35]" = torch.ops.aten.le.Scalar(alias_495, 0);  alias_495 = None
        where_75: "f32[32, 48, 35, 35]" = torch.ops.aten.where.self(le_75, full_default, getitem_422);  le_75 = getitem_422 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_153: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_75, [0, 2, 3])
        sub_396: "f32[32, 48, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_1289);  convolution_20 = unsqueeze_1289 = None
        mul_1354: "f32[32, 48, 35, 35]" = torch.ops.aten.mul.Tensor(where_75, sub_396)
        sum_154: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_1354, [0, 2, 3]);  mul_1354 = None
        mul_1355: "f32[48]" = torch.ops.aten.mul.Tensor(sum_153, 2.5510204081632654e-05)
        unsqueeze_1290: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1355, 0);  mul_1355 = None
        unsqueeze_1291: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1290, 2);  unsqueeze_1290 = None
        unsqueeze_1292: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1291, 3);  unsqueeze_1291 = None
        mul_1356: "f32[48]" = torch.ops.aten.mul.Tensor(sum_154, 2.5510204081632654e-05)
        mul_1357: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_1358: "f32[48]" = torch.ops.aten.mul.Tensor(mul_1356, mul_1357);  mul_1356 = mul_1357 = None
        unsqueeze_1293: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1358, 0);  mul_1358 = None
        unsqueeze_1294: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1293, 2);  unsqueeze_1293 = None
        unsqueeze_1295: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1294, 3);  unsqueeze_1294 = None
        mul_1359: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_62);  primals_62 = None
        unsqueeze_1296: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1359, 0);  mul_1359 = None
        unsqueeze_1297: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1296, 2);  unsqueeze_1296 = None
        unsqueeze_1298: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1297, 3);  unsqueeze_1297 = None
        mul_1360: "f32[32, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_396, unsqueeze_1295);  sub_396 = unsqueeze_1295 = None
        sub_398: "f32[32, 48, 35, 35]" = torch.ops.aten.sub.Tensor(where_75, mul_1360);  where_75 = mul_1360 = None
        sub_399: "f32[32, 48, 35, 35]" = torch.ops.aten.sub.Tensor(sub_398, unsqueeze_1292);  sub_398 = unsqueeze_1292 = None
        mul_1361: "f32[32, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_399, unsqueeze_1298);  sub_399 = unsqueeze_1298 = None
        mul_1362: "f32[48]" = torch.ops.aten.mul.Tensor(sum_154, squeeze_61);  sum_154 = squeeze_61 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_1361, cat_2, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1361 = primals_61 = None
        getitem_425: "f32[32, 288, 35, 35]" = convolution_backward_75[0]
        getitem_426: "f32[48, 288, 1, 1]" = convolution_backward_75[1];  convolution_backward_75 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_511: "f32[32, 288, 35, 35]" = torch.ops.aten.add.Tensor(add_510, getitem_425);  add_510 = getitem_425 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_76: "f32[32, 64, 35, 35]" = torch.ops.aten.where.self(le_76, full_default, slice_42);  le_76 = slice_42 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_155: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_76, [0, 2, 3])
        sub_400: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_1301);  convolution_19 = unsqueeze_1301 = None
        mul_1363: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_76, sub_400)
        sum_156: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1363, [0, 2, 3]);  mul_1363 = None
        mul_1364: "f32[64]" = torch.ops.aten.mul.Tensor(sum_155, 2.5510204081632654e-05)
        unsqueeze_1302: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1364, 0);  mul_1364 = None
        unsqueeze_1303: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1302, 2);  unsqueeze_1302 = None
        unsqueeze_1304: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1303, 3);  unsqueeze_1303 = None
        mul_1365: "f32[64]" = torch.ops.aten.mul.Tensor(sum_156, 2.5510204081632654e-05)
        mul_1366: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_1367: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1365, mul_1366);  mul_1365 = mul_1366 = None
        unsqueeze_1305: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1367, 0);  mul_1367 = None
        unsqueeze_1306: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1305, 2);  unsqueeze_1305 = None
        unsqueeze_1307: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1306, 3);  unsqueeze_1306 = None
        mul_1368: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_59);  primals_59 = None
        unsqueeze_1308: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1368, 0);  mul_1368 = None
        unsqueeze_1309: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1308, 2);  unsqueeze_1308 = None
        unsqueeze_1310: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1309, 3);  unsqueeze_1309 = None
        mul_1369: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_400, unsqueeze_1307);  sub_400 = unsqueeze_1307 = None
        sub_402: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_76, mul_1369);  where_76 = mul_1369 = None
        sub_403: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_402, unsqueeze_1304);  sub_402 = unsqueeze_1304 = None
        mul_1370: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_403, unsqueeze_1310);  sub_403 = unsqueeze_1310 = None
        mul_1371: "f32[64]" = torch.ops.aten.mul.Tensor(sum_156, squeeze_58);  sum_156 = squeeze_58 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1370, cat_2, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1370 = cat_2 = primals_58 = None
        getitem_428: "f32[32, 288, 35, 35]" = convolution_backward_76[0]
        getitem_429: "f32[64, 288, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_512: "f32[32, 288, 35, 35]" = torch.ops.aten.add.Tensor(add_511, getitem_428);  add_511 = getitem_428 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:212 in forward, code: return torch.cat(outputs, 1)
        slice_46: "f32[32, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_512, 1, 0, 64)
        slice_47: "f32[32, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_512, 1, 64, 128)
        slice_48: "f32[32, 96, 35, 35]" = torch.ops.aten.slice.Tensor(add_512, 1, 128, 224)
        slice_49: "f32[32, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_512, 1, 224, 288);  add_512 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_77: "f32[32, 64, 35, 35]" = torch.ops.aten.where.self(le_77, full_default, slice_49);  le_77 = slice_49 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_157: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_77, [0, 2, 3])
        sub_404: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_1313);  convolution_18 = unsqueeze_1313 = None
        mul_1372: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_77, sub_404)
        sum_158: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1372, [0, 2, 3]);  mul_1372 = None
        mul_1373: "f32[64]" = torch.ops.aten.mul.Tensor(sum_157, 2.5510204081632654e-05)
        unsqueeze_1314: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1373, 0);  mul_1373 = None
        unsqueeze_1315: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1314, 2);  unsqueeze_1314 = None
        unsqueeze_1316: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1315, 3);  unsqueeze_1315 = None
        mul_1374: "f32[64]" = torch.ops.aten.mul.Tensor(sum_158, 2.5510204081632654e-05)
        mul_1375: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_1376: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1374, mul_1375);  mul_1374 = mul_1375 = None
        unsqueeze_1317: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1376, 0);  mul_1376 = None
        unsqueeze_1318: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1317, 2);  unsqueeze_1317 = None
        unsqueeze_1319: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1318, 3);  unsqueeze_1318 = None
        mul_1377: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_56);  primals_56 = None
        unsqueeze_1320: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1377, 0);  mul_1377 = None
        unsqueeze_1321: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1320, 2);  unsqueeze_1320 = None
        unsqueeze_1322: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1321, 3);  unsqueeze_1321 = None
        mul_1378: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_404, unsqueeze_1319);  sub_404 = unsqueeze_1319 = None
        sub_406: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_77, mul_1378);  where_77 = mul_1378 = None
        sub_407: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_406, unsqueeze_1316);  sub_406 = unsqueeze_1316 = None
        mul_1379: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_407, unsqueeze_1322);  sub_407 = unsqueeze_1322 = None
        mul_1380: "f32[64]" = torch.ops.aten.mul.Tensor(sum_158, squeeze_55);  sum_158 = squeeze_55 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1379, avg_pool2d_1, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1379 = avg_pool2d_1 = primals_55 = None
        getitem_431: "f32[32, 256, 35, 35]" = convolution_backward_77[0]
        getitem_432: "f32[64, 256, 1, 1]" = convolution_backward_77[1];  convolution_backward_77 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:204 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward_8: "f32[32, 256, 35, 35]" = torch.ops.aten.avg_pool2d_backward.default(getitem_431, cat_1, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_431 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_78: "f32[32, 96, 35, 35]" = torch.ops.aten.where.self(le_78, full_default, slice_48);  le_78 = slice_48 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_159: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_78, [0, 2, 3])
        sub_408: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_1325);  convolution_17 = unsqueeze_1325 = None
        mul_1381: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(where_78, sub_408)
        sum_160: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1381, [0, 2, 3]);  mul_1381 = None
        mul_1382: "f32[96]" = torch.ops.aten.mul.Tensor(sum_159, 2.5510204081632654e-05)
        unsqueeze_1326: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1382, 0);  mul_1382 = None
        unsqueeze_1327: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1326, 2);  unsqueeze_1326 = None
        unsqueeze_1328: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1327, 3);  unsqueeze_1327 = None
        mul_1383: "f32[96]" = torch.ops.aten.mul.Tensor(sum_160, 2.5510204081632654e-05)
        mul_1384: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_1385: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1383, mul_1384);  mul_1383 = mul_1384 = None
        unsqueeze_1329: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1385, 0);  mul_1385 = None
        unsqueeze_1330: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1329, 2);  unsqueeze_1329 = None
        unsqueeze_1331: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1330, 3);  unsqueeze_1330 = None
        mul_1386: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_53);  primals_53 = None
        unsqueeze_1332: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1386, 0);  mul_1386 = None
        unsqueeze_1333: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1332, 2);  unsqueeze_1332 = None
        unsqueeze_1334: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1333, 3);  unsqueeze_1333 = None
        mul_1387: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_408, unsqueeze_1331);  sub_408 = unsqueeze_1331 = None
        sub_410: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(where_78, mul_1387);  where_78 = mul_1387 = None
        sub_411: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(sub_410, unsqueeze_1328);  sub_410 = unsqueeze_1328 = None
        mul_1388: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_411, unsqueeze_1334);  sub_411 = unsqueeze_1334 = None
        mul_1389: "f32[96]" = torch.ops.aten.mul.Tensor(sum_160, squeeze_52);  sum_160 = squeeze_52 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1388, relu_16, primals_52, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1388 = primals_52 = None
        getitem_434: "f32[32, 96, 35, 35]" = convolution_backward_78[0]
        getitem_435: "f32[96, 96, 3, 3]" = convolution_backward_78[1];  convolution_backward_78 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_510: "f32[32, 96, 35, 35]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
        alias_511: "f32[32, 96, 35, 35]" = torch.ops.aten.alias.default(alias_510);  alias_510 = None
        le_79: "b8[32, 96, 35, 35]" = torch.ops.aten.le.Scalar(alias_511, 0);  alias_511 = None
        where_79: "f32[32, 96, 35, 35]" = torch.ops.aten.where.self(le_79, full_default, getitem_434);  le_79 = getitem_434 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_161: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_79, [0, 2, 3])
        sub_412: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_1337);  convolution_16 = unsqueeze_1337 = None
        mul_1390: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(where_79, sub_412)
        sum_162: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1390, [0, 2, 3]);  mul_1390 = None
        mul_1391: "f32[96]" = torch.ops.aten.mul.Tensor(sum_161, 2.5510204081632654e-05)
        unsqueeze_1338: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1391, 0);  mul_1391 = None
        unsqueeze_1339: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1338, 2);  unsqueeze_1338 = None
        unsqueeze_1340: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1339, 3);  unsqueeze_1339 = None
        mul_1392: "f32[96]" = torch.ops.aten.mul.Tensor(sum_162, 2.5510204081632654e-05)
        mul_1393: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_1394: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1392, mul_1393);  mul_1392 = mul_1393 = None
        unsqueeze_1341: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1394, 0);  mul_1394 = None
        unsqueeze_1342: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1341, 2);  unsqueeze_1341 = None
        unsqueeze_1343: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1342, 3);  unsqueeze_1342 = None
        mul_1395: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_50);  primals_50 = None
        unsqueeze_1344: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1395, 0);  mul_1395 = None
        unsqueeze_1345: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1344, 2);  unsqueeze_1344 = None
        unsqueeze_1346: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1345, 3);  unsqueeze_1345 = None
        mul_1396: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_412, unsqueeze_1343);  sub_412 = unsqueeze_1343 = None
        sub_414: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(where_79, mul_1396);  where_79 = mul_1396 = None
        sub_415: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(sub_414, unsqueeze_1340);  sub_414 = unsqueeze_1340 = None
        mul_1397: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_415, unsqueeze_1346);  sub_415 = unsqueeze_1346 = None
        mul_1398: "f32[96]" = torch.ops.aten.mul.Tensor(sum_162, squeeze_49);  sum_162 = squeeze_49 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1397, relu_15, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1397 = primals_49 = None
        getitem_437: "f32[32, 64, 35, 35]" = convolution_backward_79[0]
        getitem_438: "f32[96, 64, 3, 3]" = convolution_backward_79[1];  convolution_backward_79 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_514: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
        alias_515: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(alias_514);  alias_514 = None
        le_80: "b8[32, 64, 35, 35]" = torch.ops.aten.le.Scalar(alias_515, 0);  alias_515 = None
        where_80: "f32[32, 64, 35, 35]" = torch.ops.aten.where.self(le_80, full_default, getitem_437);  le_80 = getitem_437 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_163: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_80, [0, 2, 3])
        sub_416: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_1349);  convolution_15 = unsqueeze_1349 = None
        mul_1399: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_80, sub_416)
        sum_164: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1399, [0, 2, 3]);  mul_1399 = None
        mul_1400: "f32[64]" = torch.ops.aten.mul.Tensor(sum_163, 2.5510204081632654e-05)
        unsqueeze_1350: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1400, 0);  mul_1400 = None
        unsqueeze_1351: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1350, 2);  unsqueeze_1350 = None
        unsqueeze_1352: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1351, 3);  unsqueeze_1351 = None
        mul_1401: "f32[64]" = torch.ops.aten.mul.Tensor(sum_164, 2.5510204081632654e-05)
        mul_1402: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_1403: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1401, mul_1402);  mul_1401 = mul_1402 = None
        unsqueeze_1353: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1403, 0);  mul_1403 = None
        unsqueeze_1354: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1353, 2);  unsqueeze_1353 = None
        unsqueeze_1355: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1354, 3);  unsqueeze_1354 = None
        mul_1404: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_47);  primals_47 = None
        unsqueeze_1356: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1404, 0);  mul_1404 = None
        unsqueeze_1357: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1356, 2);  unsqueeze_1356 = None
        unsqueeze_1358: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1357, 3);  unsqueeze_1357 = None
        mul_1405: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_416, unsqueeze_1355);  sub_416 = unsqueeze_1355 = None
        sub_418: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_80, mul_1405);  where_80 = mul_1405 = None
        sub_419: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_418, unsqueeze_1352);  sub_418 = unsqueeze_1352 = None
        mul_1406: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_419, unsqueeze_1358);  sub_419 = unsqueeze_1358 = None
        mul_1407: "f32[64]" = torch.ops.aten.mul.Tensor(sum_164, squeeze_46);  sum_164 = squeeze_46 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1406, cat_1, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1406 = primals_46 = None
        getitem_440: "f32[32, 256, 35, 35]" = convolution_backward_80[0]
        getitem_441: "f32[64, 256, 1, 1]" = convolution_backward_80[1];  convolution_backward_80 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_513: "f32[32, 256, 35, 35]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_8, getitem_440);  avg_pool2d_backward_8 = getitem_440 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_81: "f32[32, 64, 35, 35]" = torch.ops.aten.where.self(le_81, full_default, slice_47);  le_81 = slice_47 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_165: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_81, [0, 2, 3])
        sub_420: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_1361);  convolution_14 = unsqueeze_1361 = None
        mul_1408: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_81, sub_420)
        sum_166: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1408, [0, 2, 3]);  mul_1408 = None
        mul_1409: "f32[64]" = torch.ops.aten.mul.Tensor(sum_165, 2.5510204081632654e-05)
        unsqueeze_1362: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1409, 0);  mul_1409 = None
        unsqueeze_1363: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1362, 2);  unsqueeze_1362 = None
        unsqueeze_1364: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1363, 3);  unsqueeze_1363 = None
        mul_1410: "f32[64]" = torch.ops.aten.mul.Tensor(sum_166, 2.5510204081632654e-05)
        mul_1411: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_1412: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1410, mul_1411);  mul_1410 = mul_1411 = None
        unsqueeze_1365: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1412, 0);  mul_1412 = None
        unsqueeze_1366: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1365, 2);  unsqueeze_1365 = None
        unsqueeze_1367: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1366, 3);  unsqueeze_1366 = None
        mul_1413: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_44);  primals_44 = None
        unsqueeze_1368: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1413, 0);  mul_1413 = None
        unsqueeze_1369: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1368, 2);  unsqueeze_1368 = None
        unsqueeze_1370: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1369, 3);  unsqueeze_1369 = None
        mul_1414: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_420, unsqueeze_1367);  sub_420 = unsqueeze_1367 = None
        sub_422: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_81, mul_1414);  where_81 = mul_1414 = None
        sub_423: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_422, unsqueeze_1364);  sub_422 = unsqueeze_1364 = None
        mul_1415: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_423, unsqueeze_1370);  sub_423 = unsqueeze_1370 = None
        mul_1416: "f32[64]" = torch.ops.aten.mul.Tensor(sum_166, squeeze_43);  sum_166 = squeeze_43 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_81 = torch.ops.aten.convolution_backward.default(mul_1415, relu_13, primals_43, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1415 = primals_43 = None
        getitem_443: "f32[32, 48, 35, 35]" = convolution_backward_81[0]
        getitem_444: "f32[64, 48, 5, 5]" = convolution_backward_81[1];  convolution_backward_81 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_522: "f32[32, 48, 35, 35]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
        alias_523: "f32[32, 48, 35, 35]" = torch.ops.aten.alias.default(alias_522);  alias_522 = None
        le_82: "b8[32, 48, 35, 35]" = torch.ops.aten.le.Scalar(alias_523, 0);  alias_523 = None
        where_82: "f32[32, 48, 35, 35]" = torch.ops.aten.where.self(le_82, full_default, getitem_443);  le_82 = getitem_443 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_167: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_82, [0, 2, 3])
        sub_424: "f32[32, 48, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_1373);  convolution_13 = unsqueeze_1373 = None
        mul_1417: "f32[32, 48, 35, 35]" = torch.ops.aten.mul.Tensor(where_82, sub_424)
        sum_168: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_1417, [0, 2, 3]);  mul_1417 = None
        mul_1418: "f32[48]" = torch.ops.aten.mul.Tensor(sum_167, 2.5510204081632654e-05)
        unsqueeze_1374: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1418, 0);  mul_1418 = None
        unsqueeze_1375: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1374, 2);  unsqueeze_1374 = None
        unsqueeze_1376: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1375, 3);  unsqueeze_1375 = None
        mul_1419: "f32[48]" = torch.ops.aten.mul.Tensor(sum_168, 2.5510204081632654e-05)
        mul_1420: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_1421: "f32[48]" = torch.ops.aten.mul.Tensor(mul_1419, mul_1420);  mul_1419 = mul_1420 = None
        unsqueeze_1377: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1421, 0);  mul_1421 = None
        unsqueeze_1378: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1377, 2);  unsqueeze_1377 = None
        unsqueeze_1379: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1378, 3);  unsqueeze_1378 = None
        mul_1422: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_41);  primals_41 = None
        unsqueeze_1380: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1422, 0);  mul_1422 = None
        unsqueeze_1381: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1380, 2);  unsqueeze_1380 = None
        unsqueeze_1382: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1381, 3);  unsqueeze_1381 = None
        mul_1423: "f32[32, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_424, unsqueeze_1379);  sub_424 = unsqueeze_1379 = None
        sub_426: "f32[32, 48, 35, 35]" = torch.ops.aten.sub.Tensor(where_82, mul_1423);  where_82 = mul_1423 = None
        sub_427: "f32[32, 48, 35, 35]" = torch.ops.aten.sub.Tensor(sub_426, unsqueeze_1376);  sub_426 = unsqueeze_1376 = None
        mul_1424: "f32[32, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_427, unsqueeze_1382);  sub_427 = unsqueeze_1382 = None
        mul_1425: "f32[48]" = torch.ops.aten.mul.Tensor(sum_168, squeeze_40);  sum_168 = squeeze_40 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_82 = torch.ops.aten.convolution_backward.default(mul_1424, cat_1, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1424 = primals_40 = None
        getitem_446: "f32[32, 256, 35, 35]" = convolution_backward_82[0]
        getitem_447: "f32[48, 256, 1, 1]" = convolution_backward_82[1];  convolution_backward_82 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_514: "f32[32, 256, 35, 35]" = torch.ops.aten.add.Tensor(add_513, getitem_446);  add_513 = getitem_446 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_83: "f32[32, 64, 35, 35]" = torch.ops.aten.where.self(le_83, full_default, slice_46);  le_83 = slice_46 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_169: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_83, [0, 2, 3])
        sub_428: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_1385);  convolution_12 = unsqueeze_1385 = None
        mul_1426: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_83, sub_428)
        sum_170: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1426, [0, 2, 3]);  mul_1426 = None
        mul_1427: "f32[64]" = torch.ops.aten.mul.Tensor(sum_169, 2.5510204081632654e-05)
        unsqueeze_1386: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1427, 0);  mul_1427 = None
        unsqueeze_1387: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1386, 2);  unsqueeze_1386 = None
        unsqueeze_1388: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1387, 3);  unsqueeze_1387 = None
        mul_1428: "f32[64]" = torch.ops.aten.mul.Tensor(sum_170, 2.5510204081632654e-05)
        mul_1429: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_1430: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1428, mul_1429);  mul_1428 = mul_1429 = None
        unsqueeze_1389: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1430, 0);  mul_1430 = None
        unsqueeze_1390: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1389, 2);  unsqueeze_1389 = None
        unsqueeze_1391: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1390, 3);  unsqueeze_1390 = None
        mul_1431: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_38);  primals_38 = None
        unsqueeze_1392: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1431, 0);  mul_1431 = None
        unsqueeze_1393: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1392, 2);  unsqueeze_1392 = None
        unsqueeze_1394: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1393, 3);  unsqueeze_1393 = None
        mul_1432: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_428, unsqueeze_1391);  sub_428 = unsqueeze_1391 = None
        sub_430: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_83, mul_1432);  where_83 = mul_1432 = None
        sub_431: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_430, unsqueeze_1388);  sub_430 = unsqueeze_1388 = None
        mul_1433: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_431, unsqueeze_1394);  sub_431 = unsqueeze_1394 = None
        mul_1434: "f32[64]" = torch.ops.aten.mul.Tensor(sum_170, squeeze_37);  sum_170 = squeeze_37 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_83 = torch.ops.aten.convolution_backward.default(mul_1433, cat_1, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1433 = cat_1 = primals_37 = None
        getitem_449: "f32[32, 256, 35, 35]" = convolution_backward_83[0]
        getitem_450: "f32[64, 256, 1, 1]" = convolution_backward_83[1];  convolution_backward_83 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_515: "f32[32, 256, 35, 35]" = torch.ops.aten.add.Tensor(add_514, getitem_449);  add_514 = getitem_449 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:212 in forward, code: return torch.cat(outputs, 1)
        slice_50: "f32[32, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_515, 1, 0, 64)
        slice_51: "f32[32, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_515, 1, 64, 128)
        slice_52: "f32[32, 96, 35, 35]" = torch.ops.aten.slice.Tensor(add_515, 1, 128, 224)
        slice_53: "f32[32, 32, 35, 35]" = torch.ops.aten.slice.Tensor(add_515, 1, 224, 256);  add_515 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_84: "f32[32, 32, 35, 35]" = torch.ops.aten.where.self(le_84, full_default, slice_53);  le_84 = slice_53 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_171: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_84, [0, 2, 3])
        sub_432: "f32[32, 32, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_1397);  convolution_11 = unsqueeze_1397 = None
        mul_1435: "f32[32, 32, 35, 35]" = torch.ops.aten.mul.Tensor(where_84, sub_432)
        sum_172: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1435, [0, 2, 3]);  mul_1435 = None
        mul_1436: "f32[32]" = torch.ops.aten.mul.Tensor(sum_171, 2.5510204081632654e-05)
        unsqueeze_1398: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1436, 0);  mul_1436 = None
        unsqueeze_1399: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1398, 2);  unsqueeze_1398 = None
        unsqueeze_1400: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1399, 3);  unsqueeze_1399 = None
        mul_1437: "f32[32]" = torch.ops.aten.mul.Tensor(sum_172, 2.5510204081632654e-05)
        mul_1438: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_1439: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1437, mul_1438);  mul_1437 = mul_1438 = None
        unsqueeze_1401: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1439, 0);  mul_1439 = None
        unsqueeze_1402: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1401, 2);  unsqueeze_1401 = None
        unsqueeze_1403: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1402, 3);  unsqueeze_1402 = None
        mul_1440: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_35);  primals_35 = None
        unsqueeze_1404: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1440, 0);  mul_1440 = None
        unsqueeze_1405: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1404, 2);  unsqueeze_1404 = None
        unsqueeze_1406: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1405, 3);  unsqueeze_1405 = None
        mul_1441: "f32[32, 32, 35, 35]" = torch.ops.aten.mul.Tensor(sub_432, unsqueeze_1403);  sub_432 = unsqueeze_1403 = None
        sub_434: "f32[32, 32, 35, 35]" = torch.ops.aten.sub.Tensor(where_84, mul_1441);  where_84 = mul_1441 = None
        sub_435: "f32[32, 32, 35, 35]" = torch.ops.aten.sub.Tensor(sub_434, unsqueeze_1400);  sub_434 = unsqueeze_1400 = None
        mul_1442: "f32[32, 32, 35, 35]" = torch.ops.aten.mul.Tensor(sub_435, unsqueeze_1406);  sub_435 = unsqueeze_1406 = None
        mul_1443: "f32[32]" = torch.ops.aten.mul.Tensor(sum_172, squeeze_34);  sum_172 = squeeze_34 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_84 = torch.ops.aten.convolution_backward.default(mul_1442, avg_pool2d, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1442 = avg_pool2d = primals_34 = None
        getitem_452: "f32[32, 192, 35, 35]" = convolution_backward_84[0]
        getitem_453: "f32[32, 192, 1, 1]" = convolution_backward_84[1];  convolution_backward_84 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:204 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward_9: "f32[32, 192, 35, 35]" = torch.ops.aten.avg_pool2d_backward.default(getitem_452, getitem_12, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_452 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_85: "f32[32, 96, 35, 35]" = torch.ops.aten.where.self(le_85, full_default, slice_52);  le_85 = slice_52 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_173: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_85, [0, 2, 3])
        sub_436: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_1409);  convolution_10 = unsqueeze_1409 = None
        mul_1444: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(where_85, sub_436)
        sum_174: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1444, [0, 2, 3]);  mul_1444 = None
        mul_1445: "f32[96]" = torch.ops.aten.mul.Tensor(sum_173, 2.5510204081632654e-05)
        unsqueeze_1410: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1445, 0);  mul_1445 = None
        unsqueeze_1411: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1410, 2);  unsqueeze_1410 = None
        unsqueeze_1412: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1411, 3);  unsqueeze_1411 = None
        mul_1446: "f32[96]" = torch.ops.aten.mul.Tensor(sum_174, 2.5510204081632654e-05)
        mul_1447: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_1448: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1446, mul_1447);  mul_1446 = mul_1447 = None
        unsqueeze_1413: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1448, 0);  mul_1448 = None
        unsqueeze_1414: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1413, 2);  unsqueeze_1413 = None
        unsqueeze_1415: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1414, 3);  unsqueeze_1414 = None
        mul_1449: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_32);  primals_32 = None
        unsqueeze_1416: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1449, 0);  mul_1449 = None
        unsqueeze_1417: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1416, 2);  unsqueeze_1416 = None
        unsqueeze_1418: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1417, 3);  unsqueeze_1417 = None
        mul_1450: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_436, unsqueeze_1415);  sub_436 = unsqueeze_1415 = None
        sub_438: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(where_85, mul_1450);  where_85 = mul_1450 = None
        sub_439: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(sub_438, unsqueeze_1412);  sub_438 = unsqueeze_1412 = None
        mul_1451: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_439, unsqueeze_1418);  sub_439 = unsqueeze_1418 = None
        mul_1452: "f32[96]" = torch.ops.aten.mul.Tensor(sum_174, squeeze_31);  sum_174 = squeeze_31 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_85 = torch.ops.aten.convolution_backward.default(mul_1451, relu_9, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1451 = primals_31 = None
        getitem_455: "f32[32, 96, 35, 35]" = convolution_backward_85[0]
        getitem_456: "f32[96, 96, 3, 3]" = convolution_backward_85[1];  convolution_backward_85 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_538: "f32[32, 96, 35, 35]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
        alias_539: "f32[32, 96, 35, 35]" = torch.ops.aten.alias.default(alias_538);  alias_538 = None
        le_86: "b8[32, 96, 35, 35]" = torch.ops.aten.le.Scalar(alias_539, 0);  alias_539 = None
        where_86: "f32[32, 96, 35, 35]" = torch.ops.aten.where.self(le_86, full_default, getitem_455);  le_86 = getitem_455 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_175: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_86, [0, 2, 3])
        sub_440: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_1421);  convolution_9 = unsqueeze_1421 = None
        mul_1453: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(where_86, sub_440)
        sum_176: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1453, [0, 2, 3]);  mul_1453 = None
        mul_1454: "f32[96]" = torch.ops.aten.mul.Tensor(sum_175, 2.5510204081632654e-05)
        unsqueeze_1422: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1454, 0);  mul_1454 = None
        unsqueeze_1423: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1422, 2);  unsqueeze_1422 = None
        unsqueeze_1424: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1423, 3);  unsqueeze_1423 = None
        mul_1455: "f32[96]" = torch.ops.aten.mul.Tensor(sum_176, 2.5510204081632654e-05)
        mul_1456: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_1457: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1455, mul_1456);  mul_1455 = mul_1456 = None
        unsqueeze_1425: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1457, 0);  mul_1457 = None
        unsqueeze_1426: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1425, 2);  unsqueeze_1425 = None
        unsqueeze_1427: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1426, 3);  unsqueeze_1426 = None
        mul_1458: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_29);  primals_29 = None
        unsqueeze_1428: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1458, 0);  mul_1458 = None
        unsqueeze_1429: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1428, 2);  unsqueeze_1428 = None
        unsqueeze_1430: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1429, 3);  unsqueeze_1429 = None
        mul_1459: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_440, unsqueeze_1427);  sub_440 = unsqueeze_1427 = None
        sub_442: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(where_86, mul_1459);  where_86 = mul_1459 = None
        sub_443: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(sub_442, unsqueeze_1424);  sub_442 = unsqueeze_1424 = None
        mul_1460: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_443, unsqueeze_1430);  sub_443 = unsqueeze_1430 = None
        mul_1461: "f32[96]" = torch.ops.aten.mul.Tensor(sum_176, squeeze_28);  sum_176 = squeeze_28 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_86 = torch.ops.aten.convolution_backward.default(mul_1460, relu_8, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1460 = primals_28 = None
        getitem_458: "f32[32, 64, 35, 35]" = convolution_backward_86[0]
        getitem_459: "f32[96, 64, 3, 3]" = convolution_backward_86[1];  convolution_backward_86 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_542: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
        alias_543: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(alias_542);  alias_542 = None
        le_87: "b8[32, 64, 35, 35]" = torch.ops.aten.le.Scalar(alias_543, 0);  alias_543 = None
        where_87: "f32[32, 64, 35, 35]" = torch.ops.aten.where.self(le_87, full_default, getitem_458);  le_87 = getitem_458 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_177: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_87, [0, 2, 3])
        sub_444: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1433);  convolution_8 = unsqueeze_1433 = None
        mul_1462: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_87, sub_444)
        sum_178: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1462, [0, 2, 3]);  mul_1462 = None
        mul_1463: "f32[64]" = torch.ops.aten.mul.Tensor(sum_177, 2.5510204081632654e-05)
        unsqueeze_1434: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1463, 0);  mul_1463 = None
        unsqueeze_1435: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1434, 2);  unsqueeze_1434 = None
        unsqueeze_1436: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1435, 3);  unsqueeze_1435 = None
        mul_1464: "f32[64]" = torch.ops.aten.mul.Tensor(sum_178, 2.5510204081632654e-05)
        mul_1465: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_1466: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1464, mul_1465);  mul_1464 = mul_1465 = None
        unsqueeze_1437: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1466, 0);  mul_1466 = None
        unsqueeze_1438: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1437, 2);  unsqueeze_1437 = None
        unsqueeze_1439: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1438, 3);  unsqueeze_1438 = None
        mul_1467: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_26);  primals_26 = None
        unsqueeze_1440: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1467, 0);  mul_1467 = None
        unsqueeze_1441: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1440, 2);  unsqueeze_1440 = None
        unsqueeze_1442: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1441, 3);  unsqueeze_1441 = None
        mul_1468: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_444, unsqueeze_1439);  sub_444 = unsqueeze_1439 = None
        sub_446: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_87, mul_1468);  where_87 = mul_1468 = None
        sub_447: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_446, unsqueeze_1436);  sub_446 = unsqueeze_1436 = None
        mul_1469: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_447, unsqueeze_1442);  sub_447 = unsqueeze_1442 = None
        mul_1470: "f32[64]" = torch.ops.aten.mul.Tensor(sum_178, squeeze_25);  sum_178 = squeeze_25 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_87 = torch.ops.aten.convolution_backward.default(mul_1469, getitem_12, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1469 = primals_25 = None
        getitem_461: "f32[32, 192, 35, 35]" = convolution_backward_87[0]
        getitem_462: "f32[64, 192, 1, 1]" = convolution_backward_87[1];  convolution_backward_87 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_516: "f32[32, 192, 35, 35]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_9, getitem_461);  avg_pool2d_backward_9 = getitem_461 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_88: "f32[32, 64, 35, 35]" = torch.ops.aten.where.self(le_88, full_default, slice_51);  le_88 = slice_51 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_179: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_88, [0, 2, 3])
        sub_448: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_1445);  convolution_7 = unsqueeze_1445 = None
        mul_1471: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_88, sub_448)
        sum_180: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1471, [0, 2, 3]);  mul_1471 = None
        mul_1472: "f32[64]" = torch.ops.aten.mul.Tensor(sum_179, 2.5510204081632654e-05)
        unsqueeze_1446: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1472, 0);  mul_1472 = None
        unsqueeze_1447: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1446, 2);  unsqueeze_1446 = None
        unsqueeze_1448: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1447, 3);  unsqueeze_1447 = None
        mul_1473: "f32[64]" = torch.ops.aten.mul.Tensor(sum_180, 2.5510204081632654e-05)
        mul_1474: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_1475: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1473, mul_1474);  mul_1473 = mul_1474 = None
        unsqueeze_1449: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1475, 0);  mul_1475 = None
        unsqueeze_1450: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1449, 2);  unsqueeze_1449 = None
        unsqueeze_1451: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1450, 3);  unsqueeze_1450 = None
        mul_1476: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_23);  primals_23 = None
        unsqueeze_1452: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1476, 0);  mul_1476 = None
        unsqueeze_1453: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1452, 2);  unsqueeze_1452 = None
        unsqueeze_1454: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1453, 3);  unsqueeze_1453 = None
        mul_1477: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_448, unsqueeze_1451);  sub_448 = unsqueeze_1451 = None
        sub_450: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_88, mul_1477);  where_88 = mul_1477 = None
        sub_451: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_450, unsqueeze_1448);  sub_450 = unsqueeze_1448 = None
        mul_1478: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_451, unsqueeze_1454);  sub_451 = unsqueeze_1454 = None
        mul_1479: "f32[64]" = torch.ops.aten.mul.Tensor(sum_180, squeeze_22);  sum_180 = squeeze_22 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_88 = torch.ops.aten.convolution_backward.default(mul_1478, relu_6, primals_22, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1478 = primals_22 = None
        getitem_464: "f32[32, 48, 35, 35]" = convolution_backward_88[0]
        getitem_465: "f32[64, 48, 5, 5]" = convolution_backward_88[1];  convolution_backward_88 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_550: "f32[32, 48, 35, 35]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
        alias_551: "f32[32, 48, 35, 35]" = torch.ops.aten.alias.default(alias_550);  alias_550 = None
        le_89: "b8[32, 48, 35, 35]" = torch.ops.aten.le.Scalar(alias_551, 0);  alias_551 = None
        where_89: "f32[32, 48, 35, 35]" = torch.ops.aten.where.self(le_89, full_default, getitem_464);  le_89 = getitem_464 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_181: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_89, [0, 2, 3])
        sub_452: "f32[32, 48, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_1457);  convolution_6 = unsqueeze_1457 = None
        mul_1480: "f32[32, 48, 35, 35]" = torch.ops.aten.mul.Tensor(where_89, sub_452)
        sum_182: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_1480, [0, 2, 3]);  mul_1480 = None
        mul_1481: "f32[48]" = torch.ops.aten.mul.Tensor(sum_181, 2.5510204081632654e-05)
        unsqueeze_1458: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1481, 0);  mul_1481 = None
        unsqueeze_1459: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1458, 2);  unsqueeze_1458 = None
        unsqueeze_1460: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1459, 3);  unsqueeze_1459 = None
        mul_1482: "f32[48]" = torch.ops.aten.mul.Tensor(sum_182, 2.5510204081632654e-05)
        mul_1483: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_1484: "f32[48]" = torch.ops.aten.mul.Tensor(mul_1482, mul_1483);  mul_1482 = mul_1483 = None
        unsqueeze_1461: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1484, 0);  mul_1484 = None
        unsqueeze_1462: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1461, 2);  unsqueeze_1461 = None
        unsqueeze_1463: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1462, 3);  unsqueeze_1462 = None
        mul_1485: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_20);  primals_20 = None
        unsqueeze_1464: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1485, 0);  mul_1485 = None
        unsqueeze_1465: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1464, 2);  unsqueeze_1464 = None
        unsqueeze_1466: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1465, 3);  unsqueeze_1465 = None
        mul_1486: "f32[32, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_452, unsqueeze_1463);  sub_452 = unsqueeze_1463 = None
        sub_454: "f32[32, 48, 35, 35]" = torch.ops.aten.sub.Tensor(where_89, mul_1486);  where_89 = mul_1486 = None
        sub_455: "f32[32, 48, 35, 35]" = torch.ops.aten.sub.Tensor(sub_454, unsqueeze_1460);  sub_454 = unsqueeze_1460 = None
        mul_1487: "f32[32, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_455, unsqueeze_1466);  sub_455 = unsqueeze_1466 = None
        mul_1488: "f32[48]" = torch.ops.aten.mul.Tensor(sum_182, squeeze_19);  sum_182 = squeeze_19 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_89 = torch.ops.aten.convolution_backward.default(mul_1487, getitem_12, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1487 = primals_19 = None
        getitem_467: "f32[32, 192, 35, 35]" = convolution_backward_89[0]
        getitem_468: "f32[48, 192, 1, 1]" = convolution_backward_89[1];  convolution_backward_89 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_517: "f32[32, 192, 35, 35]" = torch.ops.aten.add.Tensor(add_516, getitem_467);  add_516 = getitem_467 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        where_90: "f32[32, 64, 35, 35]" = torch.ops.aten.where.self(le_90, full_default, slice_50);  le_90 = slice_50 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_183: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_90, [0, 2, 3])
        sub_456: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1469);  convolution_5 = unsqueeze_1469 = None
        mul_1489: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_90, sub_456)
        sum_184: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1489, [0, 2, 3]);  mul_1489 = None
        mul_1490: "f32[64]" = torch.ops.aten.mul.Tensor(sum_183, 2.5510204081632654e-05)
        unsqueeze_1470: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1490, 0);  mul_1490 = None
        unsqueeze_1471: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1470, 2);  unsqueeze_1470 = None
        unsqueeze_1472: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1471, 3);  unsqueeze_1471 = None
        mul_1491: "f32[64]" = torch.ops.aten.mul.Tensor(sum_184, 2.5510204081632654e-05)
        mul_1492: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_1493: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1491, mul_1492);  mul_1491 = mul_1492 = None
        unsqueeze_1473: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1493, 0);  mul_1493 = None
        unsqueeze_1474: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1473, 2);  unsqueeze_1473 = None
        unsqueeze_1475: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1474, 3);  unsqueeze_1474 = None
        mul_1494: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_17);  primals_17 = None
        unsqueeze_1476: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1494, 0);  mul_1494 = None
        unsqueeze_1477: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1476, 2);  unsqueeze_1476 = None
        unsqueeze_1478: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1477, 3);  unsqueeze_1477 = None
        mul_1495: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_456, unsqueeze_1475);  sub_456 = unsqueeze_1475 = None
        sub_458: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_90, mul_1495);  where_90 = mul_1495 = None
        sub_459: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_458, unsqueeze_1472);  sub_458 = unsqueeze_1472 = None
        mul_1496: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_459, unsqueeze_1478);  sub_459 = unsqueeze_1478 = None
        mul_1497: "f32[64]" = torch.ops.aten.mul.Tensor(sum_184, squeeze_16);  sum_184 = squeeze_16 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_90 = torch.ops.aten.convolution_backward.default(mul_1496, getitem_12, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1496 = getitem_12 = primals_16 = None
        getitem_470: "f32[32, 192, 35, 35]" = convolution_backward_90[0]
        getitem_471: "f32[64, 192, 1, 1]" = convolution_backward_90[1];  convolution_backward_90 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        add_518: "f32[32, 192, 35, 35]" = torch.ops.aten.add.Tensor(add_517, getitem_470);  add_517 = getitem_470 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:117 in _forward, code: x = self.maxpool2(x)
        max_pool2d_with_indices_backward_2: "f32[32, 192, 71, 71]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_518, relu_4, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_13);  add_518 = getitem_13 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_558: "f32[32, 192, 71, 71]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
        alias_559: "f32[32, 192, 71, 71]" = torch.ops.aten.alias.default(alias_558);  alias_558 = None
        le_91: "b8[32, 192, 71, 71]" = torch.ops.aten.le.Scalar(alias_559, 0);  alias_559 = None
        where_91: "f32[32, 192, 71, 71]" = torch.ops.aten.where.self(le_91, full_default, max_pool2d_with_indices_backward_2);  le_91 = max_pool2d_with_indices_backward_2 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_185: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_91, [0, 2, 3])
        sub_460: "f32[32, 192, 71, 71]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1481);  convolution_4 = unsqueeze_1481 = None
        mul_1498: "f32[32, 192, 71, 71]" = torch.ops.aten.mul.Tensor(where_91, sub_460)
        sum_186: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1498, [0, 2, 3]);  mul_1498 = None
        mul_1499: "f32[192]" = torch.ops.aten.mul.Tensor(sum_185, 6.199166831977782e-06)
        unsqueeze_1482: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1499, 0);  mul_1499 = None
        unsqueeze_1483: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1482, 2);  unsqueeze_1482 = None
        unsqueeze_1484: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1483, 3);  unsqueeze_1483 = None
        mul_1500: "f32[192]" = torch.ops.aten.mul.Tensor(sum_186, 6.199166831977782e-06)
        mul_1501: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_1502: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1500, mul_1501);  mul_1500 = mul_1501 = None
        unsqueeze_1485: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1502, 0);  mul_1502 = None
        unsqueeze_1486: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1485, 2);  unsqueeze_1485 = None
        unsqueeze_1487: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1486, 3);  unsqueeze_1486 = None
        mul_1503: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_14);  primals_14 = None
        unsqueeze_1488: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1503, 0);  mul_1503 = None
        unsqueeze_1489: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1488, 2);  unsqueeze_1488 = None
        unsqueeze_1490: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1489, 3);  unsqueeze_1489 = None
        mul_1504: "f32[32, 192, 71, 71]" = torch.ops.aten.mul.Tensor(sub_460, unsqueeze_1487);  sub_460 = unsqueeze_1487 = None
        sub_462: "f32[32, 192, 71, 71]" = torch.ops.aten.sub.Tensor(where_91, mul_1504);  where_91 = mul_1504 = None
        sub_463: "f32[32, 192, 71, 71]" = torch.ops.aten.sub.Tensor(sub_462, unsqueeze_1484);  sub_462 = unsqueeze_1484 = None
        mul_1505: "f32[32, 192, 71, 71]" = torch.ops.aten.mul.Tensor(sub_463, unsqueeze_1490);  sub_463 = unsqueeze_1490 = None
        mul_1506: "f32[192]" = torch.ops.aten.mul.Tensor(sum_186, squeeze_13);  sum_186 = squeeze_13 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_91 = torch.ops.aten.convolution_backward.default(mul_1505, relu_3, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1505 = primals_13 = None
        getitem_473: "f32[32, 80, 73, 73]" = convolution_backward_91[0]
        getitem_474: "f32[192, 80, 3, 3]" = convolution_backward_91[1];  convolution_backward_91 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_562: "f32[32, 80, 73, 73]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
        alias_563: "f32[32, 80, 73, 73]" = torch.ops.aten.alias.default(alias_562);  alias_562 = None
        le_92: "b8[32, 80, 73, 73]" = torch.ops.aten.le.Scalar(alias_563, 0);  alias_563 = None
        where_92: "f32[32, 80, 73, 73]" = torch.ops.aten.where.self(le_92, full_default, getitem_473);  le_92 = getitem_473 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_187: "f32[80]" = torch.ops.aten.sum.dim_IntList(where_92, [0, 2, 3])
        sub_464: "f32[32, 80, 73, 73]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1493);  convolution_3 = unsqueeze_1493 = None
        mul_1507: "f32[32, 80, 73, 73]" = torch.ops.aten.mul.Tensor(where_92, sub_464)
        sum_188: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_1507, [0, 2, 3]);  mul_1507 = None
        mul_1508: "f32[80]" = torch.ops.aten.mul.Tensor(sum_187, 5.864139613435917e-06)
        unsqueeze_1494: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_1508, 0);  mul_1508 = None
        unsqueeze_1495: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1494, 2);  unsqueeze_1494 = None
        unsqueeze_1496: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1495, 3);  unsqueeze_1495 = None
        mul_1509: "f32[80]" = torch.ops.aten.mul.Tensor(sum_188, 5.864139613435917e-06)
        mul_1510: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_1511: "f32[80]" = torch.ops.aten.mul.Tensor(mul_1509, mul_1510);  mul_1509 = mul_1510 = None
        unsqueeze_1497: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_1511, 0);  mul_1511 = None
        unsqueeze_1498: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1497, 2);  unsqueeze_1497 = None
        unsqueeze_1499: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1498, 3);  unsqueeze_1498 = None
        mul_1512: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_11);  primals_11 = None
        unsqueeze_1500: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_1512, 0);  mul_1512 = None
        unsqueeze_1501: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1500, 2);  unsqueeze_1500 = None
        unsqueeze_1502: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1501, 3);  unsqueeze_1501 = None
        mul_1513: "f32[32, 80, 73, 73]" = torch.ops.aten.mul.Tensor(sub_464, unsqueeze_1499);  sub_464 = unsqueeze_1499 = None
        sub_466: "f32[32, 80, 73, 73]" = torch.ops.aten.sub.Tensor(where_92, mul_1513);  where_92 = mul_1513 = None
        sub_467: "f32[32, 80, 73, 73]" = torch.ops.aten.sub.Tensor(sub_466, unsqueeze_1496);  sub_466 = unsqueeze_1496 = None
        mul_1514: "f32[32, 80, 73, 73]" = torch.ops.aten.mul.Tensor(sub_467, unsqueeze_1502);  sub_467 = unsqueeze_1502 = None
        mul_1515: "f32[80]" = torch.ops.aten.mul.Tensor(sum_188, squeeze_10);  sum_188 = squeeze_10 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_92 = torch.ops.aten.convolution_backward.default(mul_1514, getitem_6, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1514 = getitem_6 = primals_10 = None
        getitem_476: "f32[32, 64, 73, 73]" = convolution_backward_92[0]
        getitem_477: "f32[80, 64, 1, 1]" = convolution_backward_92[1];  convolution_backward_92 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:111 in _forward, code: x = self.maxpool1(x)
        max_pool2d_with_indices_backward_3: "f32[32, 64, 147, 147]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_476, relu_2, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_7);  getitem_476 = getitem_7 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_566: "f32[32, 64, 147, 147]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
        alias_567: "f32[32, 64, 147, 147]" = torch.ops.aten.alias.default(alias_566);  alias_566 = None
        le_93: "b8[32, 64, 147, 147]" = torch.ops.aten.le.Scalar(alias_567, 0);  alias_567 = None
        where_93: "f32[32, 64, 147, 147]" = torch.ops.aten.where.self(le_93, full_default, max_pool2d_with_indices_backward_3);  le_93 = max_pool2d_with_indices_backward_3 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_189: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_93, [0, 2, 3])
        sub_468: "f32[32, 64, 147, 147]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1505);  convolution_2 = unsqueeze_1505 = None
        mul_1516: "f32[32, 64, 147, 147]" = torch.ops.aten.mul.Tensor(where_93, sub_468)
        sum_190: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1516, [0, 2, 3]);  mul_1516 = None
        mul_1517: "f32[64]" = torch.ops.aten.mul.Tensor(sum_189, 1.446156693970105e-06)
        unsqueeze_1506: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1517, 0);  mul_1517 = None
        unsqueeze_1507: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1506, 2);  unsqueeze_1506 = None
        unsqueeze_1508: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1507, 3);  unsqueeze_1507 = None
        mul_1518: "f32[64]" = torch.ops.aten.mul.Tensor(sum_190, 1.446156693970105e-06)
        mul_1519: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_1520: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1518, mul_1519);  mul_1518 = mul_1519 = None
        unsqueeze_1509: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1520, 0);  mul_1520 = None
        unsqueeze_1510: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1509, 2);  unsqueeze_1509 = None
        unsqueeze_1511: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1510, 3);  unsqueeze_1510 = None
        mul_1521: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_8);  primals_8 = None
        unsqueeze_1512: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1521, 0);  mul_1521 = None
        unsqueeze_1513: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1512, 2);  unsqueeze_1512 = None
        unsqueeze_1514: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1513, 3);  unsqueeze_1513 = None
        mul_1522: "f32[32, 64, 147, 147]" = torch.ops.aten.mul.Tensor(sub_468, unsqueeze_1511);  sub_468 = unsqueeze_1511 = None
        sub_470: "f32[32, 64, 147, 147]" = torch.ops.aten.sub.Tensor(where_93, mul_1522);  where_93 = mul_1522 = None
        sub_471: "f32[32, 64, 147, 147]" = torch.ops.aten.sub.Tensor(sub_470, unsqueeze_1508);  sub_470 = unsqueeze_1508 = None
        mul_1523: "f32[32, 64, 147, 147]" = torch.ops.aten.mul.Tensor(sub_471, unsqueeze_1514);  sub_471 = unsqueeze_1514 = None
        mul_1524: "f32[64]" = torch.ops.aten.mul.Tensor(sum_190, squeeze_7);  sum_190 = squeeze_7 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_93 = torch.ops.aten.convolution_backward.default(mul_1523, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1523 = primals_7 = None
        getitem_479: "f32[32, 32, 147, 147]" = convolution_backward_93[0]
        getitem_480: "f32[64, 32, 3, 3]" = convolution_backward_93[1];  convolution_backward_93 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_570: "f32[32, 32, 147, 147]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
        alias_571: "f32[32, 32, 147, 147]" = torch.ops.aten.alias.default(alias_570);  alias_570 = None
        le_94: "b8[32, 32, 147, 147]" = torch.ops.aten.le.Scalar(alias_571, 0);  alias_571 = None
        where_94: "f32[32, 32, 147, 147]" = torch.ops.aten.where.self(le_94, full_default, getitem_479);  le_94 = getitem_479 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_191: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_94, [0, 2, 3])
        sub_472: "f32[32, 32, 147, 147]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1517);  convolution_1 = unsqueeze_1517 = None
        mul_1525: "f32[32, 32, 147, 147]" = torch.ops.aten.mul.Tensor(where_94, sub_472)
        sum_192: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1525, [0, 2, 3]);  mul_1525 = None
        mul_1526: "f32[32]" = torch.ops.aten.mul.Tensor(sum_191, 1.446156693970105e-06)
        unsqueeze_1518: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1526, 0);  mul_1526 = None
        unsqueeze_1519: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1518, 2);  unsqueeze_1518 = None
        unsqueeze_1520: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1519, 3);  unsqueeze_1519 = None
        mul_1527: "f32[32]" = torch.ops.aten.mul.Tensor(sum_192, 1.446156693970105e-06)
        mul_1528: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_1529: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1527, mul_1528);  mul_1527 = mul_1528 = None
        unsqueeze_1521: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1529, 0);  mul_1529 = None
        unsqueeze_1522: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1521, 2);  unsqueeze_1521 = None
        unsqueeze_1523: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1522, 3);  unsqueeze_1522 = None
        mul_1530: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_5);  primals_5 = None
        unsqueeze_1524: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1530, 0);  mul_1530 = None
        unsqueeze_1525: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1524, 2);  unsqueeze_1524 = None
        unsqueeze_1526: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1525, 3);  unsqueeze_1525 = None
        mul_1531: "f32[32, 32, 147, 147]" = torch.ops.aten.mul.Tensor(sub_472, unsqueeze_1523);  sub_472 = unsqueeze_1523 = None
        sub_474: "f32[32, 32, 147, 147]" = torch.ops.aten.sub.Tensor(where_94, mul_1531);  where_94 = mul_1531 = None
        sub_475: "f32[32, 32, 147, 147]" = torch.ops.aten.sub.Tensor(sub_474, unsqueeze_1520);  sub_474 = unsqueeze_1520 = None
        mul_1532: "f32[32, 32, 147, 147]" = torch.ops.aten.mul.Tensor(sub_475, unsqueeze_1526);  sub_475 = unsqueeze_1526 = None
        mul_1533: "f32[32]" = torch.ops.aten.mul.Tensor(sum_192, squeeze_4);  sum_192 = squeeze_4 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_94 = torch.ops.aten.convolution_backward.default(mul_1532, relu, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1532 = primals_4 = None
        getitem_482: "f32[32, 32, 149, 149]" = convolution_backward_94[0]
        getitem_483: "f32[32, 32, 3, 3]" = convolution_backward_94[1];  convolution_backward_94 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_574: "f32[32, 32, 149, 149]" = torch.ops.aten.alias.default(relu);  relu = None
        alias_575: "f32[32, 32, 149, 149]" = torch.ops.aten.alias.default(alias_574);  alias_574 = None
        le_95: "b8[32, 32, 149, 149]" = torch.ops.aten.le.Scalar(alias_575, 0);  alias_575 = None
        where_95: "f32[32, 32, 149, 149]" = torch.ops.aten.where.self(le_95, full_default, getitem_482);  le_95 = full_default = getitem_482 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        sum_193: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_95, [0, 2, 3])
        sub_476: "f32[32, 32, 149, 149]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1529);  convolution = unsqueeze_1529 = None
        mul_1534: "f32[32, 32, 149, 149]" = torch.ops.aten.mul.Tensor(where_95, sub_476)
        sum_194: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1534, [0, 2, 3]);  mul_1534 = None
        mul_1535: "f32[32]" = torch.ops.aten.mul.Tensor(sum_193, 1.4075942525111482e-06)
        unsqueeze_1530: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1535, 0);  mul_1535 = None
        unsqueeze_1531: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1530, 2);  unsqueeze_1530 = None
        unsqueeze_1532: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1531, 3);  unsqueeze_1531 = None
        mul_1536: "f32[32]" = torch.ops.aten.mul.Tensor(sum_194, 1.4075942525111482e-06)
        mul_1537: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_1538: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1536, mul_1537);  mul_1536 = mul_1537 = None
        unsqueeze_1533: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1538, 0);  mul_1538 = None
        unsqueeze_1534: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1533, 2);  unsqueeze_1533 = None
        unsqueeze_1535: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1534, 3);  unsqueeze_1534 = None
        mul_1539: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_2);  primals_2 = None
        unsqueeze_1536: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1539, 0);  mul_1539 = None
        unsqueeze_1537: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1536, 2);  unsqueeze_1536 = None
        unsqueeze_1538: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1537, 3);  unsqueeze_1537 = None
        mul_1540: "f32[32, 32, 149, 149]" = torch.ops.aten.mul.Tensor(sub_476, unsqueeze_1535);  sub_476 = unsqueeze_1535 = None
        sub_478: "f32[32, 32, 149, 149]" = torch.ops.aten.sub.Tensor(where_95, mul_1540);  where_95 = mul_1540 = None
        sub_479: "f32[32, 32, 149, 149]" = torch.ops.aten.sub.Tensor(sub_478, unsqueeze_1532);  sub_478 = unsqueeze_1532 = None
        mul_1541: "f32[32, 32, 149, 149]" = torch.ops.aten.mul.Tensor(sub_479, unsqueeze_1538);  sub_479 = unsqueeze_1538 = None
        mul_1542: "f32[32]" = torch.ops.aten.mul.Tensor(sum_194, squeeze_1);  sum_194 = squeeze_1 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_backward_95 = torch.ops.aten.convolution_backward.default(mul_1541, cat, primals_1, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1541 = cat = primals_1 = None
        getitem_486: "f32[32, 3, 3, 3]" = convolution_backward_95[1];  convolution_backward_95 = None
        return [getitem_486, mul_1542, sum_193, getitem_483, mul_1533, sum_191, getitem_480, mul_1524, sum_189, getitem_477, mul_1515, sum_187, getitem_474, mul_1506, sum_185, getitem_471, mul_1497, sum_183, getitem_468, mul_1488, sum_181, getitem_465, mul_1479, sum_179, getitem_462, mul_1470, sum_177, getitem_459, mul_1461, sum_175, getitem_456, mul_1452, sum_173, getitem_453, mul_1443, sum_171, getitem_450, mul_1434, sum_169, getitem_447, mul_1425, sum_167, getitem_444, mul_1416, sum_165, getitem_441, mul_1407, sum_163, getitem_438, mul_1398, sum_161, getitem_435, mul_1389, sum_159, getitem_432, mul_1380, sum_157, getitem_429, mul_1371, sum_155, getitem_426, mul_1362, sum_153, getitem_423, mul_1353, sum_151, getitem_420, mul_1344, sum_149, getitem_417, mul_1335, sum_147, getitem_414, mul_1326, sum_145, getitem_411, mul_1317, sum_143, getitem_408, mul_1308, sum_141, getitem_405, mul_1299, sum_139, getitem_402, mul_1290, sum_137, getitem_399, mul_1281, sum_135, getitem_396, mul_1272, sum_133, getitem_393, mul_1263, sum_131, getitem_390, mul_1254, sum_129, getitem_387, mul_1245, sum_127, getitem_384, mul_1236, sum_125, getitem_381, mul_1227, sum_123, getitem_378, mul_1218, sum_121, getitem_375, mul_1209, sum_119, getitem_372, mul_1200, sum_117, getitem_369, mul_1191, sum_115, getitem_366, mul_1182, sum_113, getitem_363, mul_1173, sum_111, getitem_360, mul_1164, sum_109, getitem_357, mul_1155, sum_107, getitem_354, mul_1146, sum_105, getitem_351, mul_1137, sum_103, getitem_348, mul_1128, sum_101, getitem_345, mul_1119, sum_99, getitem_342, mul_1110, sum_97, getitem_339, mul_1101, sum_95, getitem_336, mul_1092, sum_93, getitem_333, mul_1083, sum_91, getitem_330, mul_1074, sum_89, getitem_327, mul_1065, sum_87, getitem_324, mul_1056, sum_85, getitem_321, mul_1047, sum_83, getitem_318, mul_1038, sum_81, getitem_315, mul_1029, sum_79, getitem_312, mul_1020, sum_77, getitem_309, mul_1011, sum_75, getitem_306, mul_1002, sum_73, getitem_303, mul_993, sum_71, getitem_300, mul_984, sum_69, getitem_297, mul_975, sum_67, getitem_294, mul_966, sum_65, getitem_291, mul_957, sum_63, getitem_288, mul_948, sum_61, getitem_285, mul_939, sum_59, getitem_282, mul_930, sum_57, getitem_279, mul_921, sum_55, getitem_276, mul_912, sum_53, getitem_273, mul_903, sum_51, permute_9, view_4, getitem_270, mul_894, sum_48, getitem_267, mul_885, sum_46, getitem_264, mul_876, sum_44, getitem_261, mul_867, sum_42, getitem_258, mul_858, sum_40, getitem_255, mul_849, sum_38, getitem_252, mul_840, sum_36, getitem_249, mul_831, sum_34, getitem_246, mul_822, sum_32, getitem_243, mul_813, sum_30, getitem_240, mul_804, sum_28, getitem_237, mul_795, sum_26, getitem_234, mul_786, sum_24, getitem_231, mul_777, sum_22, getitem_228, mul_768, sum_20, getitem_225, mul_759, sum_18, getitem_222, mul_750, sum_16, getitem_219, mul_741, sum_14, getitem_216, mul_732, sum_12, getitem_213, mul_723, sum_10, getitem_210, mul_714, sum_8, getitem_207, mul_705, sum_6, getitem_204, mul_696, sum_4, getitem_201, mul_687, sum_2, permute_5, view_2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
        