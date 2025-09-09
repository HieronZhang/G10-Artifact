class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 3, 3, 3]", primals_2: "f32[32]", primals_3: "f32[32]", primals_4: "f32[32, 32, 3, 3]", primals_5: "f32[32]", primals_6: "f32[32]", primals_7: "f32[64, 32, 3, 3]", primals_8: "f32[64]", primals_9: "f32[64]", primals_10: "f32[80, 64, 1, 1]", primals_11: "f32[80]", primals_12: "f32[80]", primals_13: "f32[192, 80, 3, 3]", primals_14: "f32[192]", primals_15: "f32[192]", primals_16: "f32[64, 192, 1, 1]", primals_17: "f32[64]", primals_18: "f32[64]", primals_19: "f32[48, 192, 1, 1]", primals_20: "f32[48]", primals_21: "f32[48]", primals_22: "f32[64, 48, 5, 5]", primals_23: "f32[64]", primals_24: "f32[64]", primals_25: "f32[64, 192, 1, 1]", primals_26: "f32[64]", primals_27: "f32[64]", primals_28: "f32[96, 64, 3, 3]", primals_29: "f32[96]", primals_30: "f32[96]", primals_31: "f32[96, 96, 3, 3]", primals_32: "f32[96]", primals_33: "f32[96]", primals_34: "f32[32, 192, 1, 1]", primals_35: "f32[32]", primals_36: "f32[32]", primals_37: "f32[64, 256, 1, 1]", primals_38: "f32[64]", primals_39: "f32[64]", primals_40: "f32[48, 256, 1, 1]", primals_41: "f32[48]", primals_42: "f32[48]", primals_43: "f32[64, 48, 5, 5]", primals_44: "f32[64]", primals_45: "f32[64]", primals_46: "f32[64, 256, 1, 1]", primals_47: "f32[64]", primals_48: "f32[64]", primals_49: "f32[96, 64, 3, 3]", primals_50: "f32[96]", primals_51: "f32[96]", primals_52: "f32[96, 96, 3, 3]", primals_53: "f32[96]", primals_54: "f32[96]", primals_55: "f32[64, 256, 1, 1]", primals_56: "f32[64]", primals_57: "f32[64]", primals_58: "f32[64, 288, 1, 1]", primals_59: "f32[64]", primals_60: "f32[64]", primals_61: "f32[48, 288, 1, 1]", primals_62: "f32[48]", primals_63: "f32[48]", primals_64: "f32[64, 48, 5, 5]", primals_65: "f32[64]", primals_66: "f32[64]", primals_67: "f32[64, 288, 1, 1]", primals_68: "f32[64]", primals_69: "f32[64]", primals_70: "f32[96, 64, 3, 3]", primals_71: "f32[96]", primals_72: "f32[96]", primals_73: "f32[96, 96, 3, 3]", primals_74: "f32[96]", primals_75: "f32[96]", primals_76: "f32[64, 288, 1, 1]", primals_77: "f32[64]", primals_78: "f32[64]", primals_79: "f32[384, 288, 3, 3]", primals_80: "f32[384]", primals_81: "f32[384]", primals_82: "f32[64, 288, 1, 1]", primals_83: "f32[64]", primals_84: "f32[64]", primals_85: "f32[96, 64, 3, 3]", primals_86: "f32[96]", primals_87: "f32[96]", primals_88: "f32[96, 96, 3, 3]", primals_89: "f32[96]", primals_90: "f32[96]", primals_91: "f32[192, 768, 1, 1]", primals_92: "f32[192]", primals_93: "f32[192]", primals_94: "f32[128, 768, 1, 1]", primals_95: "f32[128]", primals_96: "f32[128]", primals_97: "f32[128, 128, 1, 7]", primals_98: "f32[128]", primals_99: "f32[128]", primals_100: "f32[192, 128, 7, 1]", primals_101: "f32[192]", primals_102: "f32[192]", primals_103: "f32[128, 768, 1, 1]", primals_104: "f32[128]", primals_105: "f32[128]", primals_106: "f32[128, 128, 7, 1]", primals_107: "f32[128]", primals_108: "f32[128]", primals_109: "f32[128, 128, 1, 7]", primals_110: "f32[128]", primals_111: "f32[128]", primals_112: "f32[128, 128, 7, 1]", primals_113: "f32[128]", primals_114: "f32[128]", primals_115: "f32[192, 128, 1, 7]", primals_116: "f32[192]", primals_117: "f32[192]", primals_118: "f32[192, 768, 1, 1]", primals_119: "f32[192]", primals_120: "f32[192]", primals_121: "f32[192, 768, 1, 1]", primals_122: "f32[192]", primals_123: "f32[192]", primals_124: "f32[160, 768, 1, 1]", primals_125: "f32[160]", primals_126: "f32[160]", primals_127: "f32[160, 160, 1, 7]", primals_128: "f32[160]", primals_129: "f32[160]", primals_130: "f32[192, 160, 7, 1]", primals_131: "f32[192]", primals_132: "f32[192]", primals_133: "f32[160, 768, 1, 1]", primals_134: "f32[160]", primals_135: "f32[160]", primals_136: "f32[160, 160, 7, 1]", primals_137: "f32[160]", primals_138: "f32[160]", primals_139: "f32[160, 160, 1, 7]", primals_140: "f32[160]", primals_141: "f32[160]", primals_142: "f32[160, 160, 7, 1]", primals_143: "f32[160]", primals_144: "f32[160]", primals_145: "f32[192, 160, 1, 7]", primals_146: "f32[192]", primals_147: "f32[192]", primals_148: "f32[192, 768, 1, 1]", primals_149: "f32[192]", primals_150: "f32[192]", primals_151: "f32[192, 768, 1, 1]", primals_152: "f32[192]", primals_153: "f32[192]", primals_154: "f32[160, 768, 1, 1]", primals_155: "f32[160]", primals_156: "f32[160]", primals_157: "f32[160, 160, 1, 7]", primals_158: "f32[160]", primals_159: "f32[160]", primals_160: "f32[192, 160, 7, 1]", primals_161: "f32[192]", primals_162: "f32[192]", primals_163: "f32[160, 768, 1, 1]", primals_164: "f32[160]", primals_165: "f32[160]", primals_166: "f32[160, 160, 7, 1]", primals_167: "f32[160]", primals_168: "f32[160]", primals_169: "f32[160, 160, 1, 7]", primals_170: "f32[160]", primals_171: "f32[160]", primals_172: "f32[160, 160, 7, 1]", primals_173: "f32[160]", primals_174: "f32[160]", primals_175: "f32[192, 160, 1, 7]", primals_176: "f32[192]", primals_177: "f32[192]", primals_178: "f32[192, 768, 1, 1]", primals_179: "f32[192]", primals_180: "f32[192]", primals_181: "f32[192, 768, 1, 1]", primals_182: "f32[192]", primals_183: "f32[192]", primals_184: "f32[192, 768, 1, 1]", primals_185: "f32[192]", primals_186: "f32[192]", primals_187: "f32[192, 192, 1, 7]", primals_188: "f32[192]", primals_189: "f32[192]", primals_190: "f32[192, 192, 7, 1]", primals_191: "f32[192]", primals_192: "f32[192]", primals_193: "f32[192, 768, 1, 1]", primals_194: "f32[192]", primals_195: "f32[192]", primals_196: "f32[192, 192, 7, 1]", primals_197: "f32[192]", primals_198: "f32[192]", primals_199: "f32[192, 192, 1, 7]", primals_200: "f32[192]", primals_201: "f32[192]", primals_202: "f32[192, 192, 7, 1]", primals_203: "f32[192]", primals_204: "f32[192]", primals_205: "f32[192, 192, 1, 7]", primals_206: "f32[192]", primals_207: "f32[192]", primals_208: "f32[192, 768, 1, 1]", primals_209: "f32[192]", primals_210: "f32[192]", primals_211: "f32[128, 768, 1, 1]", primals_212: "f32[128]", primals_213: "f32[128]", primals_214: "f32[768, 128, 5, 5]", primals_215: "f32[768]", primals_216: "f32[768]", primals_217: "f32[1000, 768]", primals_218: "f32[1000]", primals_219: "f32[192, 768, 1, 1]", primals_220: "f32[192]", primals_221: "f32[192]", primals_222: "f32[320, 192, 3, 3]", primals_223: "f32[320]", primals_224: "f32[320]", primals_225: "f32[192, 768, 1, 1]", primals_226: "f32[192]", primals_227: "f32[192]", primals_228: "f32[192, 192, 1, 7]", primals_229: "f32[192]", primals_230: "f32[192]", primals_231: "f32[192, 192, 7, 1]", primals_232: "f32[192]", primals_233: "f32[192]", primals_234: "f32[192, 192, 3, 3]", primals_235: "f32[192]", primals_236: "f32[192]", primals_237: "f32[320, 1280, 1, 1]", primals_238: "f32[320]", primals_239: "f32[320]", primals_240: "f32[384, 1280, 1, 1]", primals_241: "f32[384]", primals_242: "f32[384]", primals_243: "f32[384, 384, 1, 3]", primals_244: "f32[384]", primals_245: "f32[384]", primals_246: "f32[384, 384, 3, 1]", primals_247: "f32[384]", primals_248: "f32[384]", primals_249: "f32[448, 1280, 1, 1]", primals_250: "f32[448]", primals_251: "f32[448]", primals_252: "f32[384, 448, 3, 3]", primals_253: "f32[384]", primals_254: "f32[384]", primals_255: "f32[384, 384, 1, 3]", primals_256: "f32[384]", primals_257: "f32[384]", primals_258: "f32[384, 384, 3, 1]", primals_259: "f32[384]", primals_260: "f32[384]", primals_261: "f32[192, 1280, 1, 1]", primals_262: "f32[192]", primals_263: "f32[192]", primals_264: "f32[320, 2048, 1, 1]", primals_265: "f32[320]", primals_266: "f32[320]", primals_267: "f32[384, 2048, 1, 1]", primals_268: "f32[384]", primals_269: "f32[384]", primals_270: "f32[384, 384, 1, 3]", primals_271: "f32[384]", primals_272: "f32[384]", primals_273: "f32[384, 384, 3, 1]", primals_274: "f32[384]", primals_275: "f32[384]", primals_276: "f32[448, 2048, 1, 1]", primals_277: "f32[448]", primals_278: "f32[448]", primals_279: "f32[384, 448, 3, 3]", primals_280: "f32[384]", primals_281: "f32[384]", primals_282: "f32[384, 384, 1, 3]", primals_283: "f32[384]", primals_284: "f32[384]", primals_285: "f32[384, 384, 3, 1]", primals_286: "f32[384]", primals_287: "f32[384]", primals_288: "f32[192, 2048, 1, 1]", primals_289: "f32[192]", primals_290: "f32[192]", primals_291: "f32[1000, 2048]", primals_292: "f32[1000]", primals_293: "f32[32]", primals_294: "f32[32]", primals_295: "i64[]", primals_296: "f32[32]", primals_297: "f32[32]", primals_298: "i64[]", primals_299: "f32[64]", primals_300: "f32[64]", primals_301: "i64[]", primals_302: "f32[80]", primals_303: "f32[80]", primals_304: "i64[]", primals_305: "f32[192]", primals_306: "f32[192]", primals_307: "i64[]", primals_308: "f32[64]", primals_309: "f32[64]", primals_310: "i64[]", primals_311: "f32[48]", primals_312: "f32[48]", primals_313: "i64[]", primals_314: "f32[64]", primals_315: "f32[64]", primals_316: "i64[]", primals_317: "f32[64]", primals_318: "f32[64]", primals_319: "i64[]", primals_320: "f32[96]", primals_321: "f32[96]", primals_322: "i64[]", primals_323: "f32[96]", primals_324: "f32[96]", primals_325: "i64[]", primals_326: "f32[32]", primals_327: "f32[32]", primals_328: "i64[]", primals_329: "f32[64]", primals_330: "f32[64]", primals_331: "i64[]", primals_332: "f32[48]", primals_333: "f32[48]", primals_334: "i64[]", primals_335: "f32[64]", primals_336: "f32[64]", primals_337: "i64[]", primals_338: "f32[64]", primals_339: "f32[64]", primals_340: "i64[]", primals_341: "f32[96]", primals_342: "f32[96]", primals_343: "i64[]", primals_344: "f32[96]", primals_345: "f32[96]", primals_346: "i64[]", primals_347: "f32[64]", primals_348: "f32[64]", primals_349: "i64[]", primals_350: "f32[64]", primals_351: "f32[64]", primals_352: "i64[]", primals_353: "f32[48]", primals_354: "f32[48]", primals_355: "i64[]", primals_356: "f32[64]", primals_357: "f32[64]", primals_358: "i64[]", primals_359: "f32[64]", primals_360: "f32[64]", primals_361: "i64[]", primals_362: "f32[96]", primals_363: "f32[96]", primals_364: "i64[]", primals_365: "f32[96]", primals_366: "f32[96]", primals_367: "i64[]", primals_368: "f32[64]", primals_369: "f32[64]", primals_370: "i64[]", primals_371: "f32[384]", primals_372: "f32[384]", primals_373: "i64[]", primals_374: "f32[64]", primals_375: "f32[64]", primals_376: "i64[]", primals_377: "f32[96]", primals_378: "f32[96]", primals_379: "i64[]", primals_380: "f32[96]", primals_381: "f32[96]", primals_382: "i64[]", primals_383: "f32[192]", primals_384: "f32[192]", primals_385: "i64[]", primals_386: "f32[128]", primals_387: "f32[128]", primals_388: "i64[]", primals_389: "f32[128]", primals_390: "f32[128]", primals_391: "i64[]", primals_392: "f32[192]", primals_393: "f32[192]", primals_394: "i64[]", primals_395: "f32[128]", primals_396: "f32[128]", primals_397: "i64[]", primals_398: "f32[128]", primals_399: "f32[128]", primals_400: "i64[]", primals_401: "f32[128]", primals_402: "f32[128]", primals_403: "i64[]", primals_404: "f32[128]", primals_405: "f32[128]", primals_406: "i64[]", primals_407: "f32[192]", primals_408: "f32[192]", primals_409: "i64[]", primals_410: "f32[192]", primals_411: "f32[192]", primals_412: "i64[]", primals_413: "f32[192]", primals_414: "f32[192]", primals_415: "i64[]", primals_416: "f32[160]", primals_417: "f32[160]", primals_418: "i64[]", primals_419: "f32[160]", primals_420: "f32[160]", primals_421: "i64[]", primals_422: "f32[192]", primals_423: "f32[192]", primals_424: "i64[]", primals_425: "f32[160]", primals_426: "f32[160]", primals_427: "i64[]", primals_428: "f32[160]", primals_429: "f32[160]", primals_430: "i64[]", primals_431: "f32[160]", primals_432: "f32[160]", primals_433: "i64[]", primals_434: "f32[160]", primals_435: "f32[160]", primals_436: "i64[]", primals_437: "f32[192]", primals_438: "f32[192]", primals_439: "i64[]", primals_440: "f32[192]", primals_441: "f32[192]", primals_442: "i64[]", primals_443: "f32[192]", primals_444: "f32[192]", primals_445: "i64[]", primals_446: "f32[160]", primals_447: "f32[160]", primals_448: "i64[]", primals_449: "f32[160]", primals_450: "f32[160]", primals_451: "i64[]", primals_452: "f32[192]", primals_453: "f32[192]", primals_454: "i64[]", primals_455: "f32[160]", primals_456: "f32[160]", primals_457: "i64[]", primals_458: "f32[160]", primals_459: "f32[160]", primals_460: "i64[]", primals_461: "f32[160]", primals_462: "f32[160]", primals_463: "i64[]", primals_464: "f32[160]", primals_465: "f32[160]", primals_466: "i64[]", primals_467: "f32[192]", primals_468: "f32[192]", primals_469: "i64[]", primals_470: "f32[192]", primals_471: "f32[192]", primals_472: "i64[]", primals_473: "f32[192]", primals_474: "f32[192]", primals_475: "i64[]", primals_476: "f32[192]", primals_477: "f32[192]", primals_478: "i64[]", primals_479: "f32[192]", primals_480: "f32[192]", primals_481: "i64[]", primals_482: "f32[192]", primals_483: "f32[192]", primals_484: "i64[]", primals_485: "f32[192]", primals_486: "f32[192]", primals_487: "i64[]", primals_488: "f32[192]", primals_489: "f32[192]", primals_490: "i64[]", primals_491: "f32[192]", primals_492: "f32[192]", primals_493: "i64[]", primals_494: "f32[192]", primals_495: "f32[192]", primals_496: "i64[]", primals_497: "f32[192]", primals_498: "f32[192]", primals_499: "i64[]", primals_500: "f32[192]", primals_501: "f32[192]", primals_502: "i64[]", primals_503: "f32[128]", primals_504: "f32[128]", primals_505: "i64[]", primals_506: "f32[768]", primals_507: "f32[768]", primals_508: "i64[]", primals_509: "f32[192]", primals_510: "f32[192]", primals_511: "i64[]", primals_512: "f32[320]", primals_513: "f32[320]", primals_514: "i64[]", primals_515: "f32[192]", primals_516: "f32[192]", primals_517: "i64[]", primals_518: "f32[192]", primals_519: "f32[192]", primals_520: "i64[]", primals_521: "f32[192]", primals_522: "f32[192]", primals_523: "i64[]", primals_524: "f32[192]", primals_525: "f32[192]", primals_526: "i64[]", primals_527: "f32[320]", primals_528: "f32[320]", primals_529: "i64[]", primals_530: "f32[384]", primals_531: "f32[384]", primals_532: "i64[]", primals_533: "f32[384]", primals_534: "f32[384]", primals_535: "i64[]", primals_536: "f32[384]", primals_537: "f32[384]", primals_538: "i64[]", primals_539: "f32[448]", primals_540: "f32[448]", primals_541: "i64[]", primals_542: "f32[384]", primals_543: "f32[384]", primals_544: "i64[]", primals_545: "f32[384]", primals_546: "f32[384]", primals_547: "i64[]", primals_548: "f32[384]", primals_549: "f32[384]", primals_550: "i64[]", primals_551: "f32[192]", primals_552: "f32[192]", primals_553: "i64[]", primals_554: "f32[320]", primals_555: "f32[320]", primals_556: "i64[]", primals_557: "f32[384]", primals_558: "f32[384]", primals_559: "i64[]", primals_560: "f32[384]", primals_561: "f32[384]", primals_562: "i64[]", primals_563: "f32[384]", primals_564: "f32[384]", primals_565: "i64[]", primals_566: "f32[448]", primals_567: "f32[448]", primals_568: "i64[]", primals_569: "f32[384]", primals_570: "f32[384]", primals_571: "i64[]", primals_572: "f32[384]", primals_573: "f32[384]", primals_574: "i64[]", primals_575: "f32[384]", primals_576: "f32[384]", primals_577: "i64[]", primals_578: "f32[192]", primals_579: "f32[192]", primals_580: "i64[]", primals_581: "f32[32, 3, 299, 299]"):
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:97 in _transform_input, code: x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        slice_1: "f32[32, 3, 299, 299]" = torch.ops.aten.slice.Tensor(primals_581, 0, 0, 9223372036854775807);  primals_581 = None
        select: "f32[32, 299, 299]" = torch.ops.aten.select.int(slice_1, 1, 0)
        unsqueeze: "f32[32, 1, 299, 299]" = torch.ops.aten.unsqueeze.default(select, 1);  select = None
        mul: "f32[32, 1, 299, 299]" = torch.ops.aten.mul.Tensor(unsqueeze, 0.458);  unsqueeze = None
        add: "f32[32, 1, 299, 299]" = torch.ops.aten.add.Tensor(mul, -0.030000000000000027);  mul = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:98 in _transform_input, code: x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        select_1: "f32[32, 299, 299]" = torch.ops.aten.select.int(slice_1, 1, 1)
        unsqueeze_1: "f32[32, 1, 299, 299]" = torch.ops.aten.unsqueeze.default(select_1, 1);  select_1 = None
        mul_1: "f32[32, 1, 299, 299]" = torch.ops.aten.mul.Tensor(unsqueeze_1, 0.448);  unsqueeze_1 = None
        add_1: "f32[32, 1, 299, 299]" = torch.ops.aten.add.Tensor(mul_1, -0.08799999999999997);  mul_1 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:99 in _transform_input, code: x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        select_2: "f32[32, 299, 299]" = torch.ops.aten.select.int(slice_1, 1, 2);  slice_1 = None
        unsqueeze_2: "f32[32, 1, 299, 299]" = torch.ops.aten.unsqueeze.default(select_2, 1);  select_2 = None
        mul_2: "f32[32, 1, 299, 299]" = torch.ops.aten.mul.Tensor(unsqueeze_2, 0.45);  unsqueeze_2 = None
        add_2: "f32[32, 1, 299, 299]" = torch.ops.aten.add.Tensor(mul_2, -0.18799999999999994);  mul_2 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:100 in _transform_input, code: x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        cat: "f32[32, 3, 299, 299]" = torch.ops.aten.cat.default([add, add_1, add_2], 1);  add = add_1 = add_2 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution: "f32[32, 32, 149, 149]" = torch.ops.aten.convolution.default(cat, primals_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_3: "i64[]" = torch.ops.aten.add.Tensor(primals_295, 1)
        var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 32, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 32, 1, 1]" = var_mean[1];  var_mean = None
        add_4: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001)
        rsqrt: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub: "f32[32, 32, 149, 149]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul_3: "f32[32, 32, 149, 149]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_1: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
        mul_5: "f32[32]" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
        add_5: "f32[32]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        squeeze_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_6: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000014075962338);  squeeze_2 = None
        mul_7: "f32[32]" = torch.ops.aten.mul.Tensor(mul_6, 0.1);  mul_6 = None
        mul_8: "f32[32]" = torch.ops.aten.mul.Tensor(primals_294, 0.9)
        add_6: "f32[32]" = torch.ops.aten.add.Tensor(mul_7, mul_8);  mul_7 = mul_8 = None
        unsqueeze_3: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1)
        unsqueeze_4: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, -1);  unsqueeze_3 = None
        mul_9: "f32[32, 32, 149, 149]" = torch.ops.aten.mul.Tensor(mul_3, unsqueeze_4);  mul_3 = unsqueeze_4 = None
        unsqueeze_5: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
        unsqueeze_6: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, -1);  unsqueeze_5 = None
        add_7: "f32[32, 32, 149, 149]" = torch.ops.aten.add.Tensor(mul_9, unsqueeze_6);  mul_9 = unsqueeze_6 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu: "f32[32, 32, 149, 149]" = torch.ops.aten.relu.default(add_7);  add_7 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_1: "f32[32, 32, 147, 147]" = torch.ops.aten.convolution.default(relu, primals_4, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_8: "i64[]" = torch.ops.aten.add.Tensor(primals_298, 1)
        var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 32, 1, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 32, 1, 1]" = var_mean_1[1];  var_mean_1 = None
        add_9: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 0.001)
        rsqrt_1: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_1: "f32[32, 32, 147, 147]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_10: "f32[32, 32, 147, 147]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_4: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_11: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
        mul_12: "f32[32]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
        add_10: "f32[32]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        squeeze_5: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_13: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000014461587854);  squeeze_5 = None
        mul_14: "f32[32]" = torch.ops.aten.mul.Tensor(mul_13, 0.1);  mul_13 = None
        mul_15: "f32[32]" = torch.ops.aten.mul.Tensor(primals_297, 0.9)
        add_11: "f32[32]" = torch.ops.aten.add.Tensor(mul_14, mul_15);  mul_14 = mul_15 = None
        unsqueeze_7: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
        unsqueeze_8: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, -1);  unsqueeze_7 = None
        mul_16: "f32[32, 32, 147, 147]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_8);  mul_10 = unsqueeze_8 = None
        unsqueeze_9: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
        unsqueeze_10: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, -1);  unsqueeze_9 = None
        add_12: "f32[32, 32, 147, 147]" = torch.ops.aten.add.Tensor(mul_16, unsqueeze_10);  mul_16 = unsqueeze_10 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_1: "f32[32, 32, 147, 147]" = torch.ops.aten.relu.default(add_12);  add_12 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_2: "f32[32, 64, 147, 147]" = torch.ops.aten.convolution.default(relu_1, primals_7, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_13: "i64[]" = torch.ops.aten.add.Tensor(primals_301, 1)
        var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[1, 64, 1, 1]" = var_mean_2[0]
        getitem_5: "f32[1, 64, 1, 1]" = var_mean_2[1];  var_mean_2 = None
        add_14: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 0.001)
        rsqrt_2: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_2: "f32[32, 64, 147, 147]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
        mul_17: "f32[32, 64, 147, 147]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_7: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
        mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
        add_15: "f32[64]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        squeeze_8: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_20: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000014461587854);  squeeze_8 = None
        mul_21: "f32[64]" = torch.ops.aten.mul.Tensor(mul_20, 0.1);  mul_20 = None
        mul_22: "f32[64]" = torch.ops.aten.mul.Tensor(primals_300, 0.9)
        add_16: "f32[64]" = torch.ops.aten.add.Tensor(mul_21, mul_22);  mul_21 = mul_22 = None
        unsqueeze_11: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1)
        unsqueeze_12: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, -1);  unsqueeze_11 = None
        mul_23: "f32[32, 64, 147, 147]" = torch.ops.aten.mul.Tensor(mul_17, unsqueeze_12);  mul_17 = unsqueeze_12 = None
        unsqueeze_13: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1);  primals_9 = None
        unsqueeze_14: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, -1);  unsqueeze_13 = None
        add_17: "f32[32, 64, 147, 147]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_14);  mul_23 = unsqueeze_14 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_2: "f32[32, 64, 147, 147]" = torch.ops.aten.relu.default(add_17);  add_17 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:111 in _forward, code: x = self.maxpool1(x)
        max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu_2, [3, 3], [2, 2])
        getitem_6: "f32[32, 64, 73, 73]" = max_pool2d_with_indices[0]
        getitem_7: "i64[32, 64, 73, 73]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_3: "f32[32, 80, 73, 73]" = torch.ops.aten.convolution.default(getitem_6, primals_10, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_18: "i64[]" = torch.ops.aten.add.Tensor(primals_304, 1)
        var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
        getitem_8: "f32[1, 80, 1, 1]" = var_mean_3[0]
        getitem_9: "f32[1, 80, 1, 1]" = var_mean_3[1];  var_mean_3 = None
        add_19: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 0.001)
        rsqrt_3: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        sub_3: "f32[32, 80, 73, 73]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_9)
        mul_24: "f32[32, 80, 73, 73]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_10: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_25: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
        mul_26: "f32[80]" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
        add_20: "f32[80]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        squeeze_11: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_27: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000058641740017);  squeeze_11 = None
        mul_28: "f32[80]" = torch.ops.aten.mul.Tensor(mul_27, 0.1);  mul_27 = None
        mul_29: "f32[80]" = torch.ops.aten.mul.Tensor(primals_303, 0.9)
        add_21: "f32[80]" = torch.ops.aten.add.Tensor(mul_28, mul_29);  mul_28 = mul_29 = None
        unsqueeze_15: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
        unsqueeze_16: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, -1);  unsqueeze_15 = None
        mul_30: "f32[32, 80, 73, 73]" = torch.ops.aten.mul.Tensor(mul_24, unsqueeze_16);  mul_24 = unsqueeze_16 = None
        unsqueeze_17: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
        unsqueeze_18: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_17, -1);  unsqueeze_17 = None
        add_22: "f32[32, 80, 73, 73]" = torch.ops.aten.add.Tensor(mul_30, unsqueeze_18);  mul_30 = unsqueeze_18 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_3: "f32[32, 80, 73, 73]" = torch.ops.aten.relu.default(add_22);  add_22 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_4: "f32[32, 192, 71, 71]" = torch.ops.aten.convolution.default(relu_3, primals_13, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_23: "i64[]" = torch.ops.aten.add.Tensor(primals_307, 1)
        var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
        getitem_10: "f32[1, 192, 1, 1]" = var_mean_4[0]
        getitem_11: "f32[1, 192, 1, 1]" = var_mean_4[1];  var_mean_4 = None
        add_24: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 0.001)
        rsqrt_4: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_4: "f32[32, 192, 71, 71]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_11)
        mul_31: "f32[32, 192, 71, 71]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_13: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_32: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
        mul_33: "f32[192]" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
        add_25: "f32[192]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        squeeze_14: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_34: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000061992052618);  squeeze_14 = None
        mul_35: "f32[192]" = torch.ops.aten.mul.Tensor(mul_34, 0.1);  mul_34 = None
        mul_36: "f32[192]" = torch.ops.aten.mul.Tensor(primals_306, 0.9)
        add_26: "f32[192]" = torch.ops.aten.add.Tensor(mul_35, mul_36);  mul_35 = mul_36 = None
        unsqueeze_19: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1)
        unsqueeze_20: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_19, -1);  unsqueeze_19 = None
        mul_37: "f32[32, 192, 71, 71]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_20);  mul_31 = unsqueeze_20 = None
        unsqueeze_21: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1);  primals_15 = None
        unsqueeze_22: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_21, -1);  unsqueeze_21 = None
        add_27: "f32[32, 192, 71, 71]" = torch.ops.aten.add.Tensor(mul_37, unsqueeze_22);  mul_37 = unsqueeze_22 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_4: "f32[32, 192, 71, 71]" = torch.ops.aten.relu.default(add_27);  add_27 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:117 in _forward, code: x = self.maxpool2(x)
        max_pool2d_with_indices_1 = torch.ops.aten.max_pool2d_with_indices.default(relu_4, [3, 3], [2, 2])
        getitem_12: "f32[32, 192, 35, 35]" = max_pool2d_with_indices_1[0]
        getitem_13: "i64[32, 192, 35, 35]" = max_pool2d_with_indices_1[1];  max_pool2d_with_indices_1 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_5: "f32[32, 64, 35, 35]" = torch.ops.aten.convolution.default(getitem_12, primals_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_28: "i64[]" = torch.ops.aten.add.Tensor(primals_310, 1)
        var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
        getitem_14: "f32[1, 64, 1, 1]" = var_mean_5[0]
        getitem_15: "f32[1, 64, 1, 1]" = var_mean_5[1];  var_mean_5 = None
        add_29: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 0.001)
        rsqrt_5: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_5: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_15)
        mul_38: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_16: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_39: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
        mul_40: "f32[64]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
        add_30: "f32[64]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        squeeze_17: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_41: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000255108548688);  squeeze_17 = None
        mul_42: "f32[64]" = torch.ops.aten.mul.Tensor(mul_41, 0.1);  mul_41 = None
        mul_43: "f32[64]" = torch.ops.aten.mul.Tensor(primals_309, 0.9)
        add_31: "f32[64]" = torch.ops.aten.add.Tensor(mul_42, mul_43);  mul_42 = mul_43 = None
        unsqueeze_23: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
        unsqueeze_24: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_23, -1);  unsqueeze_23 = None
        mul_44: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_38, unsqueeze_24);  mul_38 = unsqueeze_24 = None
        unsqueeze_25: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
        unsqueeze_26: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_25, -1);  unsqueeze_25 = None
        add_32: "f32[32, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_26);  mul_44 = unsqueeze_26 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_5: "f32[32, 64, 35, 35]" = torch.ops.aten.relu.default(add_32);  add_32 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_6: "f32[32, 48, 35, 35]" = torch.ops.aten.convolution.default(getitem_12, primals_19, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_33: "i64[]" = torch.ops.aten.add.Tensor(primals_313, 1)
        var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
        getitem_16: "f32[1, 48, 1, 1]" = var_mean_6[0]
        getitem_17: "f32[1, 48, 1, 1]" = var_mean_6[1];  var_mean_6 = None
        add_34: "f32[1, 48, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 0.001)
        rsqrt_6: "f32[1, 48, 1, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_6: "f32[32, 48, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_17)
        mul_45: "f32[32, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_19: "f32[48]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_46: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
        mul_47: "f32[48]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
        add_35: "f32[48]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        squeeze_20: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_48: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000255108548688);  squeeze_20 = None
        mul_49: "f32[48]" = torch.ops.aten.mul.Tensor(mul_48, 0.1);  mul_48 = None
        mul_50: "f32[48]" = torch.ops.aten.mul.Tensor(primals_312, 0.9)
        add_36: "f32[48]" = torch.ops.aten.add.Tensor(mul_49, mul_50);  mul_49 = mul_50 = None
        unsqueeze_27: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1)
        unsqueeze_28: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_27, -1);  unsqueeze_27 = None
        mul_51: "f32[32, 48, 35, 35]" = torch.ops.aten.mul.Tensor(mul_45, unsqueeze_28);  mul_45 = unsqueeze_28 = None
        unsqueeze_29: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1);  primals_21 = None
        unsqueeze_30: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_29, -1);  unsqueeze_29 = None
        add_37: "f32[32, 48, 35, 35]" = torch.ops.aten.add.Tensor(mul_51, unsqueeze_30);  mul_51 = unsqueeze_30 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_6: "f32[32, 48, 35, 35]" = torch.ops.aten.relu.default(add_37);  add_37 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_7: "f32[32, 64, 35, 35]" = torch.ops.aten.convolution.default(relu_6, primals_22, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_38: "i64[]" = torch.ops.aten.add.Tensor(primals_316, 1)
        var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
        getitem_18: "f32[1, 64, 1, 1]" = var_mean_7[0]
        getitem_19: "f32[1, 64, 1, 1]" = var_mean_7[1];  var_mean_7 = None
        add_39: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 0.001)
        rsqrt_7: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_7: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_19)
        mul_52: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_22: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_53: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
        mul_54: "f32[64]" = torch.ops.aten.mul.Tensor(primals_314, 0.9)
        add_40: "f32[64]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
        squeeze_23: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_55: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000255108548688);  squeeze_23 = None
        mul_56: "f32[64]" = torch.ops.aten.mul.Tensor(mul_55, 0.1);  mul_55 = None
        mul_57: "f32[64]" = torch.ops.aten.mul.Tensor(primals_315, 0.9)
        add_41: "f32[64]" = torch.ops.aten.add.Tensor(mul_56, mul_57);  mul_56 = mul_57 = None
        unsqueeze_31: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
        unsqueeze_32: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_31, -1);  unsqueeze_31 = None
        mul_58: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_32);  mul_52 = unsqueeze_32 = None
        unsqueeze_33: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
        unsqueeze_34: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_33, -1);  unsqueeze_33 = None
        add_42: "f32[32, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_58, unsqueeze_34);  mul_58 = unsqueeze_34 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_7: "f32[32, 64, 35, 35]" = torch.ops.aten.relu.default(add_42);  add_42 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_8: "f32[32, 64, 35, 35]" = torch.ops.aten.convolution.default(getitem_12, primals_25, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_43: "i64[]" = torch.ops.aten.add.Tensor(primals_319, 1)
        var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
        getitem_20: "f32[1, 64, 1, 1]" = var_mean_8[0]
        getitem_21: "f32[1, 64, 1, 1]" = var_mean_8[1];  var_mean_8 = None
        add_44: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 0.001)
        rsqrt_8: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_8: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_21)
        mul_59: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_25: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_60: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
        mul_61: "f32[64]" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
        add_45: "f32[64]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
        squeeze_26: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_62: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000255108548688);  squeeze_26 = None
        mul_63: "f32[64]" = torch.ops.aten.mul.Tensor(mul_62, 0.1);  mul_62 = None
        mul_64: "f32[64]" = torch.ops.aten.mul.Tensor(primals_318, 0.9)
        add_46: "f32[64]" = torch.ops.aten.add.Tensor(mul_63, mul_64);  mul_63 = mul_64 = None
        unsqueeze_35: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1)
        unsqueeze_36: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_35, -1);  unsqueeze_35 = None
        mul_65: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_59, unsqueeze_36);  mul_59 = unsqueeze_36 = None
        unsqueeze_37: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1);  primals_27 = None
        unsqueeze_38: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_37, -1);  unsqueeze_37 = None
        add_47: "f32[32, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_65, unsqueeze_38);  mul_65 = unsqueeze_38 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_8: "f32[32, 64, 35, 35]" = torch.ops.aten.relu.default(add_47);  add_47 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_9: "f32[32, 96, 35, 35]" = torch.ops.aten.convolution.default(relu_8, primals_28, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_48: "i64[]" = torch.ops.aten.add.Tensor(primals_322, 1)
        var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
        getitem_22: "f32[1, 96, 1, 1]" = var_mean_9[0]
        getitem_23: "f32[1, 96, 1, 1]" = var_mean_9[1];  var_mean_9 = None
        add_49: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 0.001)
        rsqrt_9: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_9: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_23)
        mul_66: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_28: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_67: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
        mul_68: "f32[96]" = torch.ops.aten.mul.Tensor(primals_320, 0.9)
        add_50: "f32[96]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        squeeze_29: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_69: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000255108548688);  squeeze_29 = None
        mul_70: "f32[96]" = torch.ops.aten.mul.Tensor(mul_69, 0.1);  mul_69 = None
        mul_71: "f32[96]" = torch.ops.aten.mul.Tensor(primals_321, 0.9)
        add_51: "f32[96]" = torch.ops.aten.add.Tensor(mul_70, mul_71);  mul_70 = mul_71 = None
        unsqueeze_39: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
        unsqueeze_40: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_39, -1);  unsqueeze_39 = None
        mul_72: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(mul_66, unsqueeze_40);  mul_66 = unsqueeze_40 = None
        unsqueeze_41: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
        unsqueeze_42: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_41, -1);  unsqueeze_41 = None
        add_52: "f32[32, 96, 35, 35]" = torch.ops.aten.add.Tensor(mul_72, unsqueeze_42);  mul_72 = unsqueeze_42 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_9: "f32[32, 96, 35, 35]" = torch.ops.aten.relu.default(add_52);  add_52 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_10: "f32[32, 96, 35, 35]" = torch.ops.aten.convolution.default(relu_9, primals_31, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_53: "i64[]" = torch.ops.aten.add.Tensor(primals_325, 1)
        var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
        getitem_24: "f32[1, 96, 1, 1]" = var_mean_10[0]
        getitem_25: "f32[1, 96, 1, 1]" = var_mean_10[1];  var_mean_10 = None
        add_54: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 0.001)
        rsqrt_10: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_10: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_25)
        mul_73: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_31: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_74: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
        mul_75: "f32[96]" = torch.ops.aten.mul.Tensor(primals_323, 0.9)
        add_55: "f32[96]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        squeeze_32: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_76: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000255108548688);  squeeze_32 = None
        mul_77: "f32[96]" = torch.ops.aten.mul.Tensor(mul_76, 0.1);  mul_76 = None
        mul_78: "f32[96]" = torch.ops.aten.mul.Tensor(primals_324, 0.9)
        add_56: "f32[96]" = torch.ops.aten.add.Tensor(mul_77, mul_78);  mul_77 = mul_78 = None
        unsqueeze_43: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1)
        unsqueeze_44: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_43, -1);  unsqueeze_43 = None
        mul_79: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_44);  mul_73 = unsqueeze_44 = None
        unsqueeze_45: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1);  primals_33 = None
        unsqueeze_46: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_45, -1);  unsqueeze_45 = None
        add_57: "f32[32, 96, 35, 35]" = torch.ops.aten.add.Tensor(mul_79, unsqueeze_46);  mul_79 = unsqueeze_46 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_10: "f32[32, 96, 35, 35]" = torch.ops.aten.relu.default(add_57);  add_57 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:204 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d: "f32[32, 192, 35, 35]" = torch.ops.aten.avg_pool2d.default(getitem_12, [3, 3], [1, 1], [1, 1])
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_11: "f32[32, 32, 35, 35]" = torch.ops.aten.convolution.default(avg_pool2d, primals_34, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_58: "i64[]" = torch.ops.aten.add.Tensor(primals_328, 1)
        var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
        getitem_26: "f32[1, 32, 1, 1]" = var_mean_11[0]
        getitem_27: "f32[1, 32, 1, 1]" = var_mean_11[1];  var_mean_11 = None
        add_59: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 0.001)
        rsqrt_11: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        sub_11: "f32[32, 32, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_27)
        mul_80: "f32[32, 32, 35, 35]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_33: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_34: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_81: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
        mul_82: "f32[32]" = torch.ops.aten.mul.Tensor(primals_326, 0.9)
        add_60: "f32[32]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
        squeeze_35: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_83: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000255108548688);  squeeze_35 = None
        mul_84: "f32[32]" = torch.ops.aten.mul.Tensor(mul_83, 0.1);  mul_83 = None
        mul_85: "f32[32]" = torch.ops.aten.mul.Tensor(primals_327, 0.9)
        add_61: "f32[32]" = torch.ops.aten.add.Tensor(mul_84, mul_85);  mul_84 = mul_85 = None
        unsqueeze_47: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
        unsqueeze_48: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_47, -1);  unsqueeze_47 = None
        mul_86: "f32[32, 32, 35, 35]" = torch.ops.aten.mul.Tensor(mul_80, unsqueeze_48);  mul_80 = unsqueeze_48 = None
        unsqueeze_49: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
        unsqueeze_50: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_49, -1);  unsqueeze_49 = None
        add_62: "f32[32, 32, 35, 35]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_50);  mul_86 = unsqueeze_50 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_11: "f32[32, 32, 35, 35]" = torch.ops.aten.relu.default(add_62);  add_62 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:212 in forward, code: return torch.cat(outputs, 1)
        cat_1: "f32[32, 256, 35, 35]" = torch.ops.aten.cat.default([relu_5, relu_7, relu_10, relu_11], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_12: "f32[32, 64, 35, 35]" = torch.ops.aten.convolution.default(cat_1, primals_37, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_63: "i64[]" = torch.ops.aten.add.Tensor(primals_331, 1)
        var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
        getitem_28: "f32[1, 64, 1, 1]" = var_mean_12[0]
        getitem_29: "f32[1, 64, 1, 1]" = var_mean_12[1];  var_mean_12 = None
        add_64: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 0.001)
        rsqrt_12: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_12: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_29)
        mul_87: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_36: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_37: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_88: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
        mul_89: "f32[64]" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
        add_65: "f32[64]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
        squeeze_38: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_90: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000255108548688);  squeeze_38 = None
        mul_91: "f32[64]" = torch.ops.aten.mul.Tensor(mul_90, 0.1);  mul_90 = None
        mul_92: "f32[64]" = torch.ops.aten.mul.Tensor(primals_330, 0.9)
        add_66: "f32[64]" = torch.ops.aten.add.Tensor(mul_91, mul_92);  mul_91 = mul_92 = None
        unsqueeze_51: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1)
        unsqueeze_52: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_51, -1);  unsqueeze_51 = None
        mul_93: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_87, unsqueeze_52);  mul_87 = unsqueeze_52 = None
        unsqueeze_53: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1);  primals_39 = None
        unsqueeze_54: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_53, -1);  unsqueeze_53 = None
        add_67: "f32[32, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_93, unsqueeze_54);  mul_93 = unsqueeze_54 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_12: "f32[32, 64, 35, 35]" = torch.ops.aten.relu.default(add_67);  add_67 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_13: "f32[32, 48, 35, 35]" = torch.ops.aten.convolution.default(cat_1, primals_40, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_68: "i64[]" = torch.ops.aten.add.Tensor(primals_334, 1)
        var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
        getitem_30: "f32[1, 48, 1, 1]" = var_mean_13[0]
        getitem_31: "f32[1, 48, 1, 1]" = var_mean_13[1];  var_mean_13 = None
        add_69: "f32[1, 48, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 0.001)
        rsqrt_13: "f32[1, 48, 1, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_13: "f32[32, 48, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_31)
        mul_94: "f32[32, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_39: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_40: "f32[48]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_95: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
        mul_96: "f32[48]" = torch.ops.aten.mul.Tensor(primals_332, 0.9)
        add_70: "f32[48]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
        squeeze_41: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_97: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0000255108548688);  squeeze_41 = None
        mul_98: "f32[48]" = torch.ops.aten.mul.Tensor(mul_97, 0.1);  mul_97 = None
        mul_99: "f32[48]" = torch.ops.aten.mul.Tensor(primals_333, 0.9)
        add_71: "f32[48]" = torch.ops.aten.add.Tensor(mul_98, mul_99);  mul_98 = mul_99 = None
        unsqueeze_55: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
        unsqueeze_56: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_55, -1);  unsqueeze_55 = None
        mul_100: "f32[32, 48, 35, 35]" = torch.ops.aten.mul.Tensor(mul_94, unsqueeze_56);  mul_94 = unsqueeze_56 = None
        unsqueeze_57: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
        unsqueeze_58: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_57, -1);  unsqueeze_57 = None
        add_72: "f32[32, 48, 35, 35]" = torch.ops.aten.add.Tensor(mul_100, unsqueeze_58);  mul_100 = unsqueeze_58 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_13: "f32[32, 48, 35, 35]" = torch.ops.aten.relu.default(add_72);  add_72 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_14: "f32[32, 64, 35, 35]" = torch.ops.aten.convolution.default(relu_13, primals_43, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_73: "i64[]" = torch.ops.aten.add.Tensor(primals_337, 1)
        var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
        getitem_32: "f32[1, 64, 1, 1]" = var_mean_14[0]
        getitem_33: "f32[1, 64, 1, 1]" = var_mean_14[1];  var_mean_14 = None
        add_74: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 0.001)
        rsqrt_14: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_14: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_33)
        mul_101: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_42: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_43: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_102: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
        mul_103: "f32[64]" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
        add_75: "f32[64]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        squeeze_44: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_104: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0000255108548688);  squeeze_44 = None
        mul_105: "f32[64]" = torch.ops.aten.mul.Tensor(mul_104, 0.1);  mul_104 = None
        mul_106: "f32[64]" = torch.ops.aten.mul.Tensor(primals_336, 0.9)
        add_76: "f32[64]" = torch.ops.aten.add.Tensor(mul_105, mul_106);  mul_105 = mul_106 = None
        unsqueeze_59: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1)
        unsqueeze_60: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_59, -1);  unsqueeze_59 = None
        mul_107: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_101, unsqueeze_60);  mul_101 = unsqueeze_60 = None
        unsqueeze_61: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1);  primals_45 = None
        unsqueeze_62: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_61, -1);  unsqueeze_61 = None
        add_77: "f32[32, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_62);  mul_107 = unsqueeze_62 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_14: "f32[32, 64, 35, 35]" = torch.ops.aten.relu.default(add_77);  add_77 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_15: "f32[32, 64, 35, 35]" = torch.ops.aten.convolution.default(cat_1, primals_46, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_78: "i64[]" = torch.ops.aten.add.Tensor(primals_340, 1)
        var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
        getitem_34: "f32[1, 64, 1, 1]" = var_mean_15[0]
        getitem_35: "f32[1, 64, 1, 1]" = var_mean_15[1];  var_mean_15 = None
        add_79: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 0.001)
        rsqrt_15: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        sub_15: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_35)
        mul_108: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_45: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_46: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_109: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
        mul_110: "f32[64]" = torch.ops.aten.mul.Tensor(primals_338, 0.9)
        add_80: "f32[64]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
        squeeze_47: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_111: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0000255108548688);  squeeze_47 = None
        mul_112: "f32[64]" = torch.ops.aten.mul.Tensor(mul_111, 0.1);  mul_111 = None
        mul_113: "f32[64]" = torch.ops.aten.mul.Tensor(primals_339, 0.9)
        add_81: "f32[64]" = torch.ops.aten.add.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
        unsqueeze_63: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
        unsqueeze_64: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_63, -1);  unsqueeze_63 = None
        mul_114: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_108, unsqueeze_64);  mul_108 = unsqueeze_64 = None
        unsqueeze_65: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
        unsqueeze_66: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_65, -1);  unsqueeze_65 = None
        add_82: "f32[32, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_114, unsqueeze_66);  mul_114 = unsqueeze_66 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_15: "f32[32, 64, 35, 35]" = torch.ops.aten.relu.default(add_82);  add_82 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_16: "f32[32, 96, 35, 35]" = torch.ops.aten.convolution.default(relu_15, primals_49, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_83: "i64[]" = torch.ops.aten.add.Tensor(primals_343, 1)
        var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
        getitem_36: "f32[1, 96, 1, 1]" = var_mean_16[0]
        getitem_37: "f32[1, 96, 1, 1]" = var_mean_16[1];  var_mean_16 = None
        add_84: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 0.001)
        rsqrt_16: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_16: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_37)
        mul_115: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_48: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_49: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_116: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
        mul_117: "f32[96]" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
        add_85: "f32[96]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        squeeze_50: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_118: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0000255108548688);  squeeze_50 = None
        mul_119: "f32[96]" = torch.ops.aten.mul.Tensor(mul_118, 0.1);  mul_118 = None
        mul_120: "f32[96]" = torch.ops.aten.mul.Tensor(primals_342, 0.9)
        add_86: "f32[96]" = torch.ops.aten.add.Tensor(mul_119, mul_120);  mul_119 = mul_120 = None
        unsqueeze_67: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1)
        unsqueeze_68: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_67, -1);  unsqueeze_67 = None
        mul_121: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_68);  mul_115 = unsqueeze_68 = None
        unsqueeze_69: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1);  primals_51 = None
        unsqueeze_70: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_69, -1);  unsqueeze_69 = None
        add_87: "f32[32, 96, 35, 35]" = torch.ops.aten.add.Tensor(mul_121, unsqueeze_70);  mul_121 = unsqueeze_70 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_16: "f32[32, 96, 35, 35]" = torch.ops.aten.relu.default(add_87);  add_87 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_17: "f32[32, 96, 35, 35]" = torch.ops.aten.convolution.default(relu_16, primals_52, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_88: "i64[]" = torch.ops.aten.add.Tensor(primals_346, 1)
        var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
        getitem_38: "f32[1, 96, 1, 1]" = var_mean_17[0]
        getitem_39: "f32[1, 96, 1, 1]" = var_mean_17[1];  var_mean_17 = None
        add_89: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 0.001)
        rsqrt_17: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        sub_17: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_39)
        mul_122: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        squeeze_51: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_52: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_123: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
        mul_124: "f32[96]" = torch.ops.aten.mul.Tensor(primals_344, 0.9)
        add_90: "f32[96]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
        squeeze_53: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_125: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0000255108548688);  squeeze_53 = None
        mul_126: "f32[96]" = torch.ops.aten.mul.Tensor(mul_125, 0.1);  mul_125 = None
        mul_127: "f32[96]" = torch.ops.aten.mul.Tensor(primals_345, 0.9)
        add_91: "f32[96]" = torch.ops.aten.add.Tensor(mul_126, mul_127);  mul_126 = mul_127 = None
        unsqueeze_71: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
        unsqueeze_72: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_71, -1);  unsqueeze_71 = None
        mul_128: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(mul_122, unsqueeze_72);  mul_122 = unsqueeze_72 = None
        unsqueeze_73: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
        unsqueeze_74: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_73, -1);  unsqueeze_73 = None
        add_92: "f32[32, 96, 35, 35]" = torch.ops.aten.add.Tensor(mul_128, unsqueeze_74);  mul_128 = unsqueeze_74 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_17: "f32[32, 96, 35, 35]" = torch.ops.aten.relu.default(add_92);  add_92 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:204 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_1: "f32[32, 256, 35, 35]" = torch.ops.aten.avg_pool2d.default(cat_1, [3, 3], [1, 1], [1, 1])
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_18: "f32[32, 64, 35, 35]" = torch.ops.aten.convolution.default(avg_pool2d_1, primals_55, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_93: "i64[]" = torch.ops.aten.add.Tensor(primals_349, 1)
        var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
        getitem_40: "f32[1, 64, 1, 1]" = var_mean_18[0]
        getitem_41: "f32[1, 64, 1, 1]" = var_mean_18[1];  var_mean_18 = None
        add_94: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 0.001)
        rsqrt_18: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_18: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_41)
        mul_129: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        squeeze_54: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_55: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_130: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
        mul_131: "f32[64]" = torch.ops.aten.mul.Tensor(primals_347, 0.9)
        add_95: "f32[64]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
        squeeze_56: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_132: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0000255108548688);  squeeze_56 = None
        mul_133: "f32[64]" = torch.ops.aten.mul.Tensor(mul_132, 0.1);  mul_132 = None
        mul_134: "f32[64]" = torch.ops.aten.mul.Tensor(primals_348, 0.9)
        add_96: "f32[64]" = torch.ops.aten.add.Tensor(mul_133, mul_134);  mul_133 = mul_134 = None
        unsqueeze_75: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
        unsqueeze_76: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_75, -1);  unsqueeze_75 = None
        mul_135: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_129, unsqueeze_76);  mul_129 = unsqueeze_76 = None
        unsqueeze_77: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
        unsqueeze_78: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_77, -1);  unsqueeze_77 = None
        add_97: "f32[32, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_135, unsqueeze_78);  mul_135 = unsqueeze_78 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_18: "f32[32, 64, 35, 35]" = torch.ops.aten.relu.default(add_97);  add_97 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:212 in forward, code: return torch.cat(outputs, 1)
        cat_2: "f32[32, 288, 35, 35]" = torch.ops.aten.cat.default([relu_12, relu_14, relu_17, relu_18], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_19: "f32[32, 64, 35, 35]" = torch.ops.aten.convolution.default(cat_2, primals_58, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_98: "i64[]" = torch.ops.aten.add.Tensor(primals_352, 1)
        var_mean_19 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
        getitem_42: "f32[1, 64, 1, 1]" = var_mean_19[0]
        getitem_43: "f32[1, 64, 1, 1]" = var_mean_19[1];  var_mean_19 = None
        add_99: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 0.001)
        rsqrt_19: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        sub_19: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_43)
        mul_136: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        squeeze_57: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        squeeze_58: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_137: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
        mul_138: "f32[64]" = torch.ops.aten.mul.Tensor(primals_350, 0.9)
        add_100: "f32[64]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
        squeeze_59: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_139: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0000255108548688);  squeeze_59 = None
        mul_140: "f32[64]" = torch.ops.aten.mul.Tensor(mul_139, 0.1);  mul_139 = None
        mul_141: "f32[64]" = torch.ops.aten.mul.Tensor(primals_351, 0.9)
        add_101: "f32[64]" = torch.ops.aten.add.Tensor(mul_140, mul_141);  mul_140 = mul_141 = None
        unsqueeze_79: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
        unsqueeze_80: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_79, -1);  unsqueeze_79 = None
        mul_142: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_80);  mul_136 = unsqueeze_80 = None
        unsqueeze_81: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
        unsqueeze_82: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_81, -1);  unsqueeze_81 = None
        add_102: "f32[32, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_142, unsqueeze_82);  mul_142 = unsqueeze_82 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_19: "f32[32, 64, 35, 35]" = torch.ops.aten.relu.default(add_102);  add_102 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_20: "f32[32, 48, 35, 35]" = torch.ops.aten.convolution.default(cat_2, primals_61, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_103: "i64[]" = torch.ops.aten.add.Tensor(primals_355, 1)
        var_mean_20 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
        getitem_44: "f32[1, 48, 1, 1]" = var_mean_20[0]
        getitem_45: "f32[1, 48, 1, 1]" = var_mean_20[1];  var_mean_20 = None
        add_104: "f32[1, 48, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 0.001)
        rsqrt_20: "f32[1, 48, 1, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        sub_20: "f32[32, 48, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_45)
        mul_143: "f32[32, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        squeeze_60: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        squeeze_61: "f32[48]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_144: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
        mul_145: "f32[48]" = torch.ops.aten.mul.Tensor(primals_353, 0.9)
        add_105: "f32[48]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        squeeze_62: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_146: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0000255108548688);  squeeze_62 = None
        mul_147: "f32[48]" = torch.ops.aten.mul.Tensor(mul_146, 0.1);  mul_146 = None
        mul_148: "f32[48]" = torch.ops.aten.mul.Tensor(primals_354, 0.9)
        add_106: "f32[48]" = torch.ops.aten.add.Tensor(mul_147, mul_148);  mul_147 = mul_148 = None
        unsqueeze_83: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1)
        unsqueeze_84: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_83, -1);  unsqueeze_83 = None
        mul_149: "f32[32, 48, 35, 35]" = torch.ops.aten.mul.Tensor(mul_143, unsqueeze_84);  mul_143 = unsqueeze_84 = None
        unsqueeze_85: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1);  primals_63 = None
        unsqueeze_86: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_85, -1);  unsqueeze_85 = None
        add_107: "f32[32, 48, 35, 35]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_86);  mul_149 = unsqueeze_86 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_20: "f32[32, 48, 35, 35]" = torch.ops.aten.relu.default(add_107);  add_107 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_21: "f32[32, 64, 35, 35]" = torch.ops.aten.convolution.default(relu_20, primals_64, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_108: "i64[]" = torch.ops.aten.add.Tensor(primals_358, 1)
        var_mean_21 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
        getitem_46: "f32[1, 64, 1, 1]" = var_mean_21[0]
        getitem_47: "f32[1, 64, 1, 1]" = var_mean_21[1];  var_mean_21 = None
        add_109: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 0.001)
        rsqrt_21: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        sub_21: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_47)
        mul_150: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        squeeze_63: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_64: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_151: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
        mul_152: "f32[64]" = torch.ops.aten.mul.Tensor(primals_356, 0.9)
        add_110: "f32[64]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
        squeeze_65: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_153: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0000255108548688);  squeeze_65 = None
        mul_154: "f32[64]" = torch.ops.aten.mul.Tensor(mul_153, 0.1);  mul_153 = None
        mul_155: "f32[64]" = torch.ops.aten.mul.Tensor(primals_357, 0.9)
        add_111: "f32[64]" = torch.ops.aten.add.Tensor(mul_154, mul_155);  mul_154 = mul_155 = None
        unsqueeze_87: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
        unsqueeze_88: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_87, -1);  unsqueeze_87 = None
        mul_156: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_150, unsqueeze_88);  mul_150 = unsqueeze_88 = None
        unsqueeze_89: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
        unsqueeze_90: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_89, -1);  unsqueeze_89 = None
        add_112: "f32[32, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_156, unsqueeze_90);  mul_156 = unsqueeze_90 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_21: "f32[32, 64, 35, 35]" = torch.ops.aten.relu.default(add_112);  add_112 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_22: "f32[32, 64, 35, 35]" = torch.ops.aten.convolution.default(cat_2, primals_67, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_113: "i64[]" = torch.ops.aten.add.Tensor(primals_361, 1)
        var_mean_22 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
        getitem_48: "f32[1, 64, 1, 1]" = var_mean_22[0]
        getitem_49: "f32[1, 64, 1, 1]" = var_mean_22[1];  var_mean_22 = None
        add_114: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 0.001)
        rsqrt_22: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        sub_22: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_49)
        mul_157: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        squeeze_66: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        squeeze_67: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_158: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
        mul_159: "f32[64]" = torch.ops.aten.mul.Tensor(primals_359, 0.9)
        add_115: "f32[64]" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
        squeeze_68: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_160: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0000255108548688);  squeeze_68 = None
        mul_161: "f32[64]" = torch.ops.aten.mul.Tensor(mul_160, 0.1);  mul_160 = None
        mul_162: "f32[64]" = torch.ops.aten.mul.Tensor(primals_360, 0.9)
        add_116: "f32[64]" = torch.ops.aten.add.Tensor(mul_161, mul_162);  mul_161 = mul_162 = None
        unsqueeze_91: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1)
        unsqueeze_92: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_91, -1);  unsqueeze_91 = None
        mul_163: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_92);  mul_157 = unsqueeze_92 = None
        unsqueeze_93: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1);  primals_69 = None
        unsqueeze_94: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_93, -1);  unsqueeze_93 = None
        add_117: "f32[32, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_163, unsqueeze_94);  mul_163 = unsqueeze_94 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_22: "f32[32, 64, 35, 35]" = torch.ops.aten.relu.default(add_117);  add_117 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_23: "f32[32, 96, 35, 35]" = torch.ops.aten.convolution.default(relu_22, primals_70, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_118: "i64[]" = torch.ops.aten.add.Tensor(primals_364, 1)
        var_mean_23 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
        getitem_50: "f32[1, 96, 1, 1]" = var_mean_23[0]
        getitem_51: "f32[1, 96, 1, 1]" = var_mean_23[1];  var_mean_23 = None
        add_119: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 0.001)
        rsqrt_23: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        sub_23: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_51)
        mul_164: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        squeeze_69: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        squeeze_70: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_165: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
        mul_166: "f32[96]" = torch.ops.aten.mul.Tensor(primals_362, 0.9)
        add_120: "f32[96]" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
        squeeze_71: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_167: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0000255108548688);  squeeze_71 = None
        mul_168: "f32[96]" = torch.ops.aten.mul.Tensor(mul_167, 0.1);  mul_167 = None
        mul_169: "f32[96]" = torch.ops.aten.mul.Tensor(primals_363, 0.9)
        add_121: "f32[96]" = torch.ops.aten.add.Tensor(mul_168, mul_169);  mul_168 = mul_169 = None
        unsqueeze_95: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
        unsqueeze_96: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_95, -1);  unsqueeze_95 = None
        mul_170: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(mul_164, unsqueeze_96);  mul_164 = unsqueeze_96 = None
        unsqueeze_97: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
        unsqueeze_98: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_97, -1);  unsqueeze_97 = None
        add_122: "f32[32, 96, 35, 35]" = torch.ops.aten.add.Tensor(mul_170, unsqueeze_98);  mul_170 = unsqueeze_98 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_23: "f32[32, 96, 35, 35]" = torch.ops.aten.relu.default(add_122);  add_122 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_24: "f32[32, 96, 35, 35]" = torch.ops.aten.convolution.default(relu_23, primals_73, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_123: "i64[]" = torch.ops.aten.add.Tensor(primals_367, 1)
        var_mean_24 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
        getitem_52: "f32[1, 96, 1, 1]" = var_mean_24[0]
        getitem_53: "f32[1, 96, 1, 1]" = var_mean_24[1];  var_mean_24 = None
        add_124: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 0.001)
        rsqrt_24: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
        sub_24: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_53)
        mul_171: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        squeeze_72: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_73: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_172: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
        mul_173: "f32[96]" = torch.ops.aten.mul.Tensor(primals_365, 0.9)
        add_125: "f32[96]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
        squeeze_74: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_174: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0000255108548688);  squeeze_74 = None
        mul_175: "f32[96]" = torch.ops.aten.mul.Tensor(mul_174, 0.1);  mul_174 = None
        mul_176: "f32[96]" = torch.ops.aten.mul.Tensor(primals_366, 0.9)
        add_126: "f32[96]" = torch.ops.aten.add.Tensor(mul_175, mul_176);  mul_175 = mul_176 = None
        unsqueeze_99: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1)
        unsqueeze_100: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_99, -1);  unsqueeze_99 = None
        mul_177: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(mul_171, unsqueeze_100);  mul_171 = unsqueeze_100 = None
        unsqueeze_101: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1);  primals_75 = None
        unsqueeze_102: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_101, -1);  unsqueeze_101 = None
        add_127: "f32[32, 96, 35, 35]" = torch.ops.aten.add.Tensor(mul_177, unsqueeze_102);  mul_177 = unsqueeze_102 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_24: "f32[32, 96, 35, 35]" = torch.ops.aten.relu.default(add_127);  add_127 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:204 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_2: "f32[32, 288, 35, 35]" = torch.ops.aten.avg_pool2d.default(cat_2, [3, 3], [1, 1], [1, 1])
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_25: "f32[32, 64, 35, 35]" = torch.ops.aten.convolution.default(avg_pool2d_2, primals_76, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_128: "i64[]" = torch.ops.aten.add.Tensor(primals_370, 1)
        var_mean_25 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
        getitem_54: "f32[1, 64, 1, 1]" = var_mean_25[0]
        getitem_55: "f32[1, 64, 1, 1]" = var_mean_25[1];  var_mean_25 = None
        add_129: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 0.001)
        rsqrt_25: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        sub_25: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_55)
        mul_178: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        squeeze_75: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        squeeze_76: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_179: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
        mul_180: "f32[64]" = torch.ops.aten.mul.Tensor(primals_368, 0.9)
        add_130: "f32[64]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
        squeeze_77: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_181: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0000255108548688);  squeeze_77 = None
        mul_182: "f32[64]" = torch.ops.aten.mul.Tensor(mul_181, 0.1);  mul_181 = None
        mul_183: "f32[64]" = torch.ops.aten.mul.Tensor(primals_369, 0.9)
        add_131: "f32[64]" = torch.ops.aten.add.Tensor(mul_182, mul_183);  mul_182 = mul_183 = None
        unsqueeze_103: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
        unsqueeze_104: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_103, -1);  unsqueeze_103 = None
        mul_184: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_178, unsqueeze_104);  mul_178 = unsqueeze_104 = None
        unsqueeze_105: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
        unsqueeze_106: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_105, -1);  unsqueeze_105 = None
        add_132: "f32[32, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_184, unsqueeze_106);  mul_184 = unsqueeze_106 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_25: "f32[32, 64, 35, 35]" = torch.ops.aten.relu.default(add_132);  add_132 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:212 in forward, code: return torch.cat(outputs, 1)
        cat_3: "f32[32, 288, 35, 35]" = torch.ops.aten.cat.default([relu_19, relu_21, relu_24, relu_25], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_26: "f32[32, 384, 17, 17]" = torch.ops.aten.convolution.default(cat_3, primals_79, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_133: "i64[]" = torch.ops.aten.add.Tensor(primals_373, 1)
        var_mean_26 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
        getitem_56: "f32[1, 384, 1, 1]" = var_mean_26[0]
        getitem_57: "f32[1, 384, 1, 1]" = var_mean_26[1];  var_mean_26 = None
        add_134: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 0.001)
        rsqrt_26: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_26: "f32[32, 384, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_57)
        mul_185: "f32[32, 384, 17, 17]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        squeeze_78: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        squeeze_79: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_186: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
        mul_187: "f32[384]" = torch.ops.aten.mul.Tensor(primals_371, 0.9)
        add_135: "f32[384]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
        squeeze_80: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        mul_188: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0001081431815724);  squeeze_80 = None
        mul_189: "f32[384]" = torch.ops.aten.mul.Tensor(mul_188, 0.1);  mul_188 = None
        mul_190: "f32[384]" = torch.ops.aten.mul.Tensor(primals_372, 0.9)
        add_136: "f32[384]" = torch.ops.aten.add.Tensor(mul_189, mul_190);  mul_189 = mul_190 = None
        unsqueeze_107: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1)
        unsqueeze_108: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_107, -1);  unsqueeze_107 = None
        mul_191: "f32[32, 384, 17, 17]" = torch.ops.aten.mul.Tensor(mul_185, unsqueeze_108);  mul_185 = unsqueeze_108 = None
        unsqueeze_109: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1);  primals_81 = None
        unsqueeze_110: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_109, -1);  unsqueeze_109 = None
        add_137: "f32[32, 384, 17, 17]" = torch.ops.aten.add.Tensor(mul_191, unsqueeze_110);  mul_191 = unsqueeze_110 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_26: "f32[32, 384, 17, 17]" = torch.ops.aten.relu.default(add_137);  add_137 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_27: "f32[32, 64, 35, 35]" = torch.ops.aten.convolution.default(cat_3, primals_82, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_138: "i64[]" = torch.ops.aten.add.Tensor(primals_376, 1)
        var_mean_27 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
        getitem_58: "f32[1, 64, 1, 1]" = var_mean_27[0]
        getitem_59: "f32[1, 64, 1, 1]" = var_mean_27[1];  var_mean_27 = None
        add_139: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 0.001)
        rsqrt_27: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
        sub_27: "f32[32, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_59)
        mul_192: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        squeeze_81: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        squeeze_82: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_193: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
        mul_194: "f32[64]" = torch.ops.aten.mul.Tensor(primals_374, 0.9)
        add_140: "f32[64]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
        squeeze_83: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_195: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0000255108548688);  squeeze_83 = None
        mul_196: "f32[64]" = torch.ops.aten.mul.Tensor(mul_195, 0.1);  mul_195 = None
        mul_197: "f32[64]" = torch.ops.aten.mul.Tensor(primals_375, 0.9)
        add_141: "f32[64]" = torch.ops.aten.add.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None
        unsqueeze_111: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
        unsqueeze_112: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, -1);  unsqueeze_111 = None
        mul_198: "f32[32, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_192, unsqueeze_112);  mul_192 = unsqueeze_112 = None
        unsqueeze_113: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
        unsqueeze_114: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_113, -1);  unsqueeze_113 = None
        add_142: "f32[32, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_198, unsqueeze_114);  mul_198 = unsqueeze_114 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_27: "f32[32, 64, 35, 35]" = torch.ops.aten.relu.default(add_142);  add_142 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_28: "f32[32, 96, 35, 35]" = torch.ops.aten.convolution.default(relu_27, primals_85, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_143: "i64[]" = torch.ops.aten.add.Tensor(primals_379, 1)
        var_mean_28 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
        getitem_60: "f32[1, 96, 1, 1]" = var_mean_28[0]
        getitem_61: "f32[1, 96, 1, 1]" = var_mean_28[1];  var_mean_28 = None
        add_144: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 0.001)
        rsqrt_28: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        sub_28: "f32[32, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_61)
        mul_199: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        squeeze_84: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        squeeze_85: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_200: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
        mul_201: "f32[96]" = torch.ops.aten.mul.Tensor(primals_377, 0.9)
        add_145: "f32[96]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
        squeeze_86: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_202: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0000255108548688);  squeeze_86 = None
        mul_203: "f32[96]" = torch.ops.aten.mul.Tensor(mul_202, 0.1);  mul_202 = None
        mul_204: "f32[96]" = torch.ops.aten.mul.Tensor(primals_378, 0.9)
        add_146: "f32[96]" = torch.ops.aten.add.Tensor(mul_203, mul_204);  mul_203 = mul_204 = None
        unsqueeze_115: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1)
        unsqueeze_116: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_115, -1);  unsqueeze_115 = None
        mul_205: "f32[32, 96, 35, 35]" = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_116);  mul_199 = unsqueeze_116 = None
        unsqueeze_117: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1);  primals_87 = None
        unsqueeze_118: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, -1);  unsqueeze_117 = None
        add_147: "f32[32, 96, 35, 35]" = torch.ops.aten.add.Tensor(mul_205, unsqueeze_118);  mul_205 = unsqueeze_118 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_28: "f32[32, 96, 35, 35]" = torch.ops.aten.relu.default(add_147);  add_147 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_29: "f32[32, 96, 17, 17]" = torch.ops.aten.convolution.default(relu_28, primals_88, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_148: "i64[]" = torch.ops.aten.add.Tensor(primals_382, 1)
        var_mean_29 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
        getitem_62: "f32[1, 96, 1, 1]" = var_mean_29[0]
        getitem_63: "f32[1, 96, 1, 1]" = var_mean_29[1];  var_mean_29 = None
        add_149: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 0.001)
        rsqrt_29: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        sub_29: "f32[32, 96, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_63)
        mul_206: "f32[32, 96, 17, 17]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        squeeze_87: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        squeeze_88: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_207: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
        mul_208: "f32[96]" = torch.ops.aten.mul.Tensor(primals_380, 0.9)
        add_150: "f32[96]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
        squeeze_89: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_209: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0001081431815724);  squeeze_89 = None
        mul_210: "f32[96]" = torch.ops.aten.mul.Tensor(mul_209, 0.1);  mul_209 = None
        mul_211: "f32[96]" = torch.ops.aten.mul.Tensor(primals_381, 0.9)
        add_151: "f32[96]" = torch.ops.aten.add.Tensor(mul_210, mul_211);  mul_210 = mul_211 = None
        unsqueeze_119: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
        unsqueeze_120: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_119, -1);  unsqueeze_119 = None
        mul_212: "f32[32, 96, 17, 17]" = torch.ops.aten.mul.Tensor(mul_206, unsqueeze_120);  mul_206 = unsqueeze_120 = None
        unsqueeze_121: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
        unsqueeze_122: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, -1);  unsqueeze_121 = None
        add_152: "f32[32, 96, 17, 17]" = torch.ops.aten.add.Tensor(mul_212, unsqueeze_122);  mul_212 = unsqueeze_122 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_29: "f32[32, 96, 17, 17]" = torch.ops.aten.relu.default(add_152);  add_152 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:233 in _forward, code: branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        max_pool2d_with_indices_2 = torch.ops.aten.max_pool2d_with_indices.default(cat_3, [3, 3], [2, 2])
        getitem_64: "f32[32, 288, 17, 17]" = max_pool2d_with_indices_2[0]
        getitem_65: "i64[32, 288, 17, 17]" = max_pool2d_with_indices_2[1];  max_pool2d_with_indices_2 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:240 in forward, code: return torch.cat(outputs, 1)
        cat_4: "f32[32, 768, 17, 17]" = torch.ops.aten.cat.default([relu_26, relu_29, getitem_64], 1);  getitem_64 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_30: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(cat_4, primals_91, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_153: "i64[]" = torch.ops.aten.add.Tensor(primals_385, 1)
        var_mean_30 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
        getitem_66: "f32[1, 192, 1, 1]" = var_mean_30[0]
        getitem_67: "f32[1, 192, 1, 1]" = var_mean_30[1];  var_mean_30 = None
        add_154: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 0.001)
        rsqrt_30: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        sub_30: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_67)
        mul_213: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        squeeze_90: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        squeeze_91: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_214: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
        mul_215: "f32[192]" = torch.ops.aten.mul.Tensor(primals_383, 0.9)
        add_155: "f32[192]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
        squeeze_92: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_216: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0001081431815724);  squeeze_92 = None
        mul_217: "f32[192]" = torch.ops.aten.mul.Tensor(mul_216, 0.1);  mul_216 = None
        mul_218: "f32[192]" = torch.ops.aten.mul.Tensor(primals_384, 0.9)
        add_156: "f32[192]" = torch.ops.aten.add.Tensor(mul_217, mul_218);  mul_217 = mul_218 = None
        unsqueeze_123: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1)
        unsqueeze_124: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_123, -1);  unsqueeze_123 = None
        mul_219: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_213, unsqueeze_124);  mul_213 = unsqueeze_124 = None
        unsqueeze_125: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1);  primals_93 = None
        unsqueeze_126: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_125, -1);  unsqueeze_125 = None
        add_157: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_219, unsqueeze_126);  mul_219 = unsqueeze_126 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_30: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_157);  add_157 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_31: "f32[32, 128, 17, 17]" = torch.ops.aten.convolution.default(cat_4, primals_94, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_158: "i64[]" = torch.ops.aten.add.Tensor(primals_388, 1)
        var_mean_31 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
        getitem_68: "f32[1, 128, 1, 1]" = var_mean_31[0]
        getitem_69: "f32[1, 128, 1, 1]" = var_mean_31[1];  var_mean_31 = None
        add_159: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 0.001)
        rsqrt_31: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        sub_31: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_69)
        mul_220: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        squeeze_93: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        squeeze_94: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_221: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
        mul_222: "f32[128]" = torch.ops.aten.mul.Tensor(primals_386, 0.9)
        add_160: "f32[128]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
        squeeze_95: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_223: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0001081431815724);  squeeze_95 = None
        mul_224: "f32[128]" = torch.ops.aten.mul.Tensor(mul_223, 0.1);  mul_223 = None
        mul_225: "f32[128]" = torch.ops.aten.mul.Tensor(primals_387, 0.9)
        add_161: "f32[128]" = torch.ops.aten.add.Tensor(mul_224, mul_225);  mul_224 = mul_225 = None
        unsqueeze_127: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
        unsqueeze_128: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, -1);  unsqueeze_127 = None
        mul_226: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_128);  mul_220 = unsqueeze_128 = None
        unsqueeze_129: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
        unsqueeze_130: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, -1);  unsqueeze_129 = None
        add_162: "f32[32, 128, 17, 17]" = torch.ops.aten.add.Tensor(mul_226, unsqueeze_130);  mul_226 = unsqueeze_130 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_31: "f32[32, 128, 17, 17]" = torch.ops.aten.relu.default(add_162);  add_162 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_32: "f32[32, 128, 17, 17]" = torch.ops.aten.convolution.default(relu_31, primals_97, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_163: "i64[]" = torch.ops.aten.add.Tensor(primals_391, 1)
        var_mean_32 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
        getitem_70: "f32[1, 128, 1, 1]" = var_mean_32[0]
        getitem_71: "f32[1, 128, 1, 1]" = var_mean_32[1];  var_mean_32 = None
        add_164: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 0.001)
        rsqrt_32: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
        sub_32: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_71)
        mul_227: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        squeeze_96: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        squeeze_97: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_228: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
        mul_229: "f32[128]" = torch.ops.aten.mul.Tensor(primals_389, 0.9)
        add_165: "f32[128]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
        squeeze_98: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_230: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0001081431815724);  squeeze_98 = None
        mul_231: "f32[128]" = torch.ops.aten.mul.Tensor(mul_230, 0.1);  mul_230 = None
        mul_232: "f32[128]" = torch.ops.aten.mul.Tensor(primals_390, 0.9)
        add_166: "f32[128]" = torch.ops.aten.add.Tensor(mul_231, mul_232);  mul_231 = mul_232 = None
        unsqueeze_131: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
        unsqueeze_132: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, -1);  unsqueeze_131 = None
        mul_233: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(mul_227, unsqueeze_132);  mul_227 = unsqueeze_132 = None
        unsqueeze_133: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
        unsqueeze_134: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, -1);  unsqueeze_133 = None
        add_167: "f32[32, 128, 17, 17]" = torch.ops.aten.add.Tensor(mul_233, unsqueeze_134);  mul_233 = unsqueeze_134 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_32: "f32[32, 128, 17, 17]" = torch.ops.aten.relu.default(add_167);  add_167 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_33: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_32, primals_100, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_168: "i64[]" = torch.ops.aten.add.Tensor(primals_394, 1)
        var_mean_33 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
        getitem_72: "f32[1, 192, 1, 1]" = var_mean_33[0]
        getitem_73: "f32[1, 192, 1, 1]" = var_mean_33[1];  var_mean_33 = None
        add_169: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 0.001)
        rsqrt_33: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
        sub_33: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_73)
        mul_234: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        squeeze_99: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        squeeze_100: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_235: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
        mul_236: "f32[192]" = torch.ops.aten.mul.Tensor(primals_392, 0.9)
        add_170: "f32[192]" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
        squeeze_101: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        mul_237: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0001081431815724);  squeeze_101 = None
        mul_238: "f32[192]" = torch.ops.aten.mul.Tensor(mul_237, 0.1);  mul_237 = None
        mul_239: "f32[192]" = torch.ops.aten.mul.Tensor(primals_393, 0.9)
        add_171: "f32[192]" = torch.ops.aten.add.Tensor(mul_238, mul_239);  mul_238 = mul_239 = None
        unsqueeze_135: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
        unsqueeze_136: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, -1);  unsqueeze_135 = None
        mul_240: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_234, unsqueeze_136);  mul_234 = unsqueeze_136 = None
        unsqueeze_137: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
        unsqueeze_138: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_137, -1);  unsqueeze_137 = None
        add_172: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_240, unsqueeze_138);  mul_240 = unsqueeze_138 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_33: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_172);  add_172 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_34: "f32[32, 128, 17, 17]" = torch.ops.aten.convolution.default(cat_4, primals_103, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_173: "i64[]" = torch.ops.aten.add.Tensor(primals_397, 1)
        var_mean_34 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
        getitem_74: "f32[1, 128, 1, 1]" = var_mean_34[0]
        getitem_75: "f32[1, 128, 1, 1]" = var_mean_34[1];  var_mean_34 = None
        add_174: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 0.001)
        rsqrt_34: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_34: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_75)
        mul_241: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        squeeze_102: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        squeeze_103: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_242: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
        mul_243: "f32[128]" = torch.ops.aten.mul.Tensor(primals_395, 0.9)
        add_175: "f32[128]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
        squeeze_104: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_244: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0001081431815724);  squeeze_104 = None
        mul_245: "f32[128]" = torch.ops.aten.mul.Tensor(mul_244, 0.1);  mul_244 = None
        mul_246: "f32[128]" = torch.ops.aten.mul.Tensor(primals_396, 0.9)
        add_176: "f32[128]" = torch.ops.aten.add.Tensor(mul_245, mul_246);  mul_245 = mul_246 = None
        unsqueeze_139: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1)
        unsqueeze_140: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, -1);  unsqueeze_139 = None
        mul_247: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(mul_241, unsqueeze_140);  mul_241 = unsqueeze_140 = None
        unsqueeze_141: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1);  primals_105 = None
        unsqueeze_142: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, -1);  unsqueeze_141 = None
        add_177: "f32[32, 128, 17, 17]" = torch.ops.aten.add.Tensor(mul_247, unsqueeze_142);  mul_247 = unsqueeze_142 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_34: "f32[32, 128, 17, 17]" = torch.ops.aten.relu.default(add_177);  add_177 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_35: "f32[32, 128, 17, 17]" = torch.ops.aten.convolution.default(relu_34, primals_106, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_178: "i64[]" = torch.ops.aten.add.Tensor(primals_400, 1)
        var_mean_35 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
        getitem_76: "f32[1, 128, 1, 1]" = var_mean_35[0]
        getitem_77: "f32[1, 128, 1, 1]" = var_mean_35[1];  var_mean_35 = None
        add_179: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 0.001)
        rsqrt_35: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
        sub_35: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_77)
        mul_248: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        squeeze_105: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_106: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
        mul_249: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
        mul_250: "f32[128]" = torch.ops.aten.mul.Tensor(primals_398, 0.9)
        add_180: "f32[128]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
        squeeze_107: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_251: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0001081431815724);  squeeze_107 = None
        mul_252: "f32[128]" = torch.ops.aten.mul.Tensor(mul_251, 0.1);  mul_251 = None
        mul_253: "f32[128]" = torch.ops.aten.mul.Tensor(primals_399, 0.9)
        add_181: "f32[128]" = torch.ops.aten.add.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
        unsqueeze_143: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
        unsqueeze_144: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, -1);  unsqueeze_143 = None
        mul_254: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(mul_248, unsqueeze_144);  mul_248 = unsqueeze_144 = None
        unsqueeze_145: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
        unsqueeze_146: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, -1);  unsqueeze_145 = None
        add_182: "f32[32, 128, 17, 17]" = torch.ops.aten.add.Tensor(mul_254, unsqueeze_146);  mul_254 = unsqueeze_146 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_35: "f32[32, 128, 17, 17]" = torch.ops.aten.relu.default(add_182);  add_182 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_36: "f32[32, 128, 17, 17]" = torch.ops.aten.convolution.default(relu_35, primals_109, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_183: "i64[]" = torch.ops.aten.add.Tensor(primals_403, 1)
        var_mean_36 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
        getitem_78: "f32[1, 128, 1, 1]" = var_mean_36[0]
        getitem_79: "f32[1, 128, 1, 1]" = var_mean_36[1];  var_mean_36 = None
        add_184: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 0.001)
        rsqrt_36: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
        sub_36: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_79)
        mul_255: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        squeeze_108: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        squeeze_109: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_256: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
        mul_257: "f32[128]" = torch.ops.aten.mul.Tensor(primals_401, 0.9)
        add_185: "f32[128]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
        squeeze_110: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_258: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0001081431815724);  squeeze_110 = None
        mul_259: "f32[128]" = torch.ops.aten.mul.Tensor(mul_258, 0.1);  mul_258 = None
        mul_260: "f32[128]" = torch.ops.aten.mul.Tensor(primals_402, 0.9)
        add_186: "f32[128]" = torch.ops.aten.add.Tensor(mul_259, mul_260);  mul_259 = mul_260 = None
        unsqueeze_147: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1)
        unsqueeze_148: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_147, -1);  unsqueeze_147 = None
        mul_261: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(mul_255, unsqueeze_148);  mul_255 = unsqueeze_148 = None
        unsqueeze_149: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1);  primals_111 = None
        unsqueeze_150: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, -1);  unsqueeze_149 = None
        add_187: "f32[32, 128, 17, 17]" = torch.ops.aten.add.Tensor(mul_261, unsqueeze_150);  mul_261 = unsqueeze_150 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_36: "f32[32, 128, 17, 17]" = torch.ops.aten.relu.default(add_187);  add_187 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_37: "f32[32, 128, 17, 17]" = torch.ops.aten.convolution.default(relu_36, primals_112, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_188: "i64[]" = torch.ops.aten.add.Tensor(primals_406, 1)
        var_mean_37 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
        getitem_80: "f32[1, 128, 1, 1]" = var_mean_37[0]
        getitem_81: "f32[1, 128, 1, 1]" = var_mean_37[1];  var_mean_37 = None
        add_189: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 0.001)
        rsqrt_37: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
        sub_37: "f32[32, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_81)
        mul_262: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        squeeze_111: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        squeeze_112: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_263: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
        mul_264: "f32[128]" = torch.ops.aten.mul.Tensor(primals_404, 0.9)
        add_190: "f32[128]" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
        squeeze_113: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
        mul_265: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0001081431815724);  squeeze_113 = None
        mul_266: "f32[128]" = torch.ops.aten.mul.Tensor(mul_265, 0.1);  mul_265 = None
        mul_267: "f32[128]" = torch.ops.aten.mul.Tensor(primals_405, 0.9)
        add_191: "f32[128]" = torch.ops.aten.add.Tensor(mul_266, mul_267);  mul_266 = mul_267 = None
        unsqueeze_151: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
        unsqueeze_152: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, -1);  unsqueeze_151 = None
        mul_268: "f32[32, 128, 17, 17]" = torch.ops.aten.mul.Tensor(mul_262, unsqueeze_152);  mul_262 = unsqueeze_152 = None
        unsqueeze_153: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
        unsqueeze_154: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, -1);  unsqueeze_153 = None
        add_192: "f32[32, 128, 17, 17]" = torch.ops.aten.add.Tensor(mul_268, unsqueeze_154);  mul_268 = unsqueeze_154 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_37: "f32[32, 128, 17, 17]" = torch.ops.aten.relu.default(add_192);  add_192 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_38: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_37, primals_115, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_193: "i64[]" = torch.ops.aten.add.Tensor(primals_409, 1)
        var_mean_38 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
        getitem_82: "f32[1, 192, 1, 1]" = var_mean_38[0]
        getitem_83: "f32[1, 192, 1, 1]" = var_mean_38[1];  var_mean_38 = None
        add_194: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 0.001)
        rsqrt_38: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
        sub_38: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_83)
        mul_269: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        squeeze_114: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        squeeze_115: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_270: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
        mul_271: "f32[192]" = torch.ops.aten.mul.Tensor(primals_407, 0.9)
        add_195: "f32[192]" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
        squeeze_116: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        mul_272: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0001081431815724);  squeeze_116 = None
        mul_273: "f32[192]" = torch.ops.aten.mul.Tensor(mul_272, 0.1);  mul_272 = None
        mul_274: "f32[192]" = torch.ops.aten.mul.Tensor(primals_408, 0.9)
        add_196: "f32[192]" = torch.ops.aten.add.Tensor(mul_273, mul_274);  mul_273 = mul_274 = None
        unsqueeze_155: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1)
        unsqueeze_156: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, -1);  unsqueeze_155 = None
        mul_275: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_269, unsqueeze_156);  mul_269 = unsqueeze_156 = None
        unsqueeze_157: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1);  primals_117 = None
        unsqueeze_158: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, -1);  unsqueeze_157 = None
        add_197: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_275, unsqueeze_158);  mul_275 = unsqueeze_158 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_38: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_197);  add_197 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:278 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_3: "f32[32, 768, 17, 17]" = torch.ops.aten.avg_pool2d.default(cat_4, [3, 3], [1, 1], [1, 1])
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_39: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(avg_pool2d_3, primals_118, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_198: "i64[]" = torch.ops.aten.add.Tensor(primals_412, 1)
        var_mean_39 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
        getitem_84: "f32[1, 192, 1, 1]" = var_mean_39[0]
        getitem_85: "f32[1, 192, 1, 1]" = var_mean_39[1];  var_mean_39 = None
        add_199: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 0.001)
        rsqrt_39: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
        sub_39: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_85)
        mul_276: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        squeeze_117: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        squeeze_118: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_277: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
        mul_278: "f32[192]" = torch.ops.aten.mul.Tensor(primals_410, 0.9)
        add_200: "f32[192]" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
        squeeze_119: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
        mul_279: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0001081431815724);  squeeze_119 = None
        mul_280: "f32[192]" = torch.ops.aten.mul.Tensor(mul_279, 0.1);  mul_279 = None
        mul_281: "f32[192]" = torch.ops.aten.mul.Tensor(primals_411, 0.9)
        add_201: "f32[192]" = torch.ops.aten.add.Tensor(mul_280, mul_281);  mul_280 = mul_281 = None
        unsqueeze_159: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
        unsqueeze_160: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, -1);  unsqueeze_159 = None
        mul_282: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_276, unsqueeze_160);  mul_276 = unsqueeze_160 = None
        unsqueeze_161: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
        unsqueeze_162: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, -1);  unsqueeze_161 = None
        add_202: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_282, unsqueeze_162);  mul_282 = unsqueeze_162 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_39: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_202);  add_202 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:286 in forward, code: return torch.cat(outputs, 1)
        cat_5: "f32[32, 768, 17, 17]" = torch.ops.aten.cat.default([relu_30, relu_33, relu_38, relu_39], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_40: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(cat_5, primals_121, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_203: "i64[]" = torch.ops.aten.add.Tensor(primals_415, 1)
        var_mean_40 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
        getitem_86: "f32[1, 192, 1, 1]" = var_mean_40[0]
        getitem_87: "f32[1, 192, 1, 1]" = var_mean_40[1];  var_mean_40 = None
        add_204: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 0.001)
        rsqrt_40: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
        sub_40: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_87)
        mul_283: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        squeeze_120: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        squeeze_121: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_284: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
        mul_285: "f32[192]" = torch.ops.aten.mul.Tensor(primals_413, 0.9)
        add_205: "f32[192]" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
        squeeze_122: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
        mul_286: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0001081431815724);  squeeze_122 = None
        mul_287: "f32[192]" = torch.ops.aten.mul.Tensor(mul_286, 0.1);  mul_286 = None
        mul_288: "f32[192]" = torch.ops.aten.mul.Tensor(primals_414, 0.9)
        add_206: "f32[192]" = torch.ops.aten.add.Tensor(mul_287, mul_288);  mul_287 = mul_288 = None
        unsqueeze_163: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1)
        unsqueeze_164: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, -1);  unsqueeze_163 = None
        mul_289: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_283, unsqueeze_164);  mul_283 = unsqueeze_164 = None
        unsqueeze_165: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1);  primals_123 = None
        unsqueeze_166: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, -1);  unsqueeze_165 = None
        add_207: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_289, unsqueeze_166);  mul_289 = unsqueeze_166 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_40: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_207);  add_207 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_41: "f32[32, 160, 17, 17]" = torch.ops.aten.convolution.default(cat_5, primals_124, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_208: "i64[]" = torch.ops.aten.add.Tensor(primals_418, 1)
        var_mean_41 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
        getitem_88: "f32[1, 160, 1, 1]" = var_mean_41[0]
        getitem_89: "f32[1, 160, 1, 1]" = var_mean_41[1];  var_mean_41 = None
        add_209: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 0.001)
        rsqrt_41: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
        sub_41: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_89)
        mul_290: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
        squeeze_123: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
        squeeze_124: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_291: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
        mul_292: "f32[160]" = torch.ops.aten.mul.Tensor(primals_416, 0.9)
        add_210: "f32[160]" = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
        squeeze_125: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
        mul_293: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0001081431815724);  squeeze_125 = None
        mul_294: "f32[160]" = torch.ops.aten.mul.Tensor(mul_293, 0.1);  mul_293 = None
        mul_295: "f32[160]" = torch.ops.aten.mul.Tensor(primals_417, 0.9)
        add_211: "f32[160]" = torch.ops.aten.add.Tensor(mul_294, mul_295);  mul_294 = mul_295 = None
        unsqueeze_167: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1)
        unsqueeze_168: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, -1);  unsqueeze_167 = None
        mul_296: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_290, unsqueeze_168);  mul_290 = unsqueeze_168 = None
        unsqueeze_169: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
        unsqueeze_170: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, -1);  unsqueeze_169 = None
        add_212: "f32[32, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_296, unsqueeze_170);  mul_296 = unsqueeze_170 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_41: "f32[32, 160, 17, 17]" = torch.ops.aten.relu.default(add_212);  add_212 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_42: "f32[32, 160, 17, 17]" = torch.ops.aten.convolution.default(relu_41, primals_127, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_213: "i64[]" = torch.ops.aten.add.Tensor(primals_421, 1)
        var_mean_42 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
        getitem_90: "f32[1, 160, 1, 1]" = var_mean_42[0]
        getitem_91: "f32[1, 160, 1, 1]" = var_mean_42[1];  var_mean_42 = None
        add_214: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 0.001)
        rsqrt_42: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
        sub_42: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_91)
        mul_297: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        squeeze_126: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        squeeze_127: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_298: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
        mul_299: "f32[160]" = torch.ops.aten.mul.Tensor(primals_419, 0.9)
        add_215: "f32[160]" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
        squeeze_128: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
        mul_300: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0001081431815724);  squeeze_128 = None
        mul_301: "f32[160]" = torch.ops.aten.mul.Tensor(mul_300, 0.1);  mul_300 = None
        mul_302: "f32[160]" = torch.ops.aten.mul.Tensor(primals_420, 0.9)
        add_216: "f32[160]" = torch.ops.aten.add.Tensor(mul_301, mul_302);  mul_301 = mul_302 = None
        unsqueeze_171: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1)
        unsqueeze_172: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, -1);  unsqueeze_171 = None
        mul_303: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_297, unsqueeze_172);  mul_297 = unsqueeze_172 = None
        unsqueeze_173: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1);  primals_129 = None
        unsqueeze_174: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, -1);  unsqueeze_173 = None
        add_217: "f32[32, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_303, unsqueeze_174);  mul_303 = unsqueeze_174 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_42: "f32[32, 160, 17, 17]" = torch.ops.aten.relu.default(add_217);  add_217 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_43: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_42, primals_130, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_218: "i64[]" = torch.ops.aten.add.Tensor(primals_424, 1)
        var_mean_43 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
        getitem_92: "f32[1, 192, 1, 1]" = var_mean_43[0]
        getitem_93: "f32[1, 192, 1, 1]" = var_mean_43[1];  var_mean_43 = None
        add_219: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 0.001)
        rsqrt_43: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
        sub_43: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_93)
        mul_304: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        squeeze_129: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        squeeze_130: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
        mul_305: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
        mul_306: "f32[192]" = torch.ops.aten.mul.Tensor(primals_422, 0.9)
        add_220: "f32[192]" = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
        squeeze_131: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
        mul_307: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0001081431815724);  squeeze_131 = None
        mul_308: "f32[192]" = torch.ops.aten.mul.Tensor(mul_307, 0.1);  mul_307 = None
        mul_309: "f32[192]" = torch.ops.aten.mul.Tensor(primals_423, 0.9)
        add_221: "f32[192]" = torch.ops.aten.add.Tensor(mul_308, mul_309);  mul_308 = mul_309 = None
        unsqueeze_175: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1)
        unsqueeze_176: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, -1);  unsqueeze_175 = None
        mul_310: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_304, unsqueeze_176);  mul_304 = unsqueeze_176 = None
        unsqueeze_177: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_132, -1);  primals_132 = None
        unsqueeze_178: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, -1);  unsqueeze_177 = None
        add_222: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_310, unsqueeze_178);  mul_310 = unsqueeze_178 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_43: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_222);  add_222 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_44: "f32[32, 160, 17, 17]" = torch.ops.aten.convolution.default(cat_5, primals_133, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_223: "i64[]" = torch.ops.aten.add.Tensor(primals_427, 1)
        var_mean_44 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
        getitem_94: "f32[1, 160, 1, 1]" = var_mean_44[0]
        getitem_95: "f32[1, 160, 1, 1]" = var_mean_44[1];  var_mean_44 = None
        add_224: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 0.001)
        rsqrt_44: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
        sub_44: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_95)
        mul_311: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
        squeeze_132: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        squeeze_133: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
        mul_312: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
        mul_313: "f32[160]" = torch.ops.aten.mul.Tensor(primals_425, 0.9)
        add_225: "f32[160]" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
        squeeze_134: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
        mul_314: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0001081431815724);  squeeze_134 = None
        mul_315: "f32[160]" = torch.ops.aten.mul.Tensor(mul_314, 0.1);  mul_314 = None
        mul_316: "f32[160]" = torch.ops.aten.mul.Tensor(primals_426, 0.9)
        add_226: "f32[160]" = torch.ops.aten.add.Tensor(mul_315, mul_316);  mul_315 = mul_316 = None
        unsqueeze_179: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1)
        unsqueeze_180: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, -1);  unsqueeze_179 = None
        mul_317: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_311, unsqueeze_180);  mul_311 = unsqueeze_180 = None
        unsqueeze_181: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_135, -1);  primals_135 = None
        unsqueeze_182: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, -1);  unsqueeze_181 = None
        add_227: "f32[32, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_317, unsqueeze_182);  mul_317 = unsqueeze_182 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_44: "f32[32, 160, 17, 17]" = torch.ops.aten.relu.default(add_227);  add_227 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_45: "f32[32, 160, 17, 17]" = torch.ops.aten.convolution.default(relu_44, primals_136, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_228: "i64[]" = torch.ops.aten.add.Tensor(primals_430, 1)
        var_mean_45 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
        getitem_96: "f32[1, 160, 1, 1]" = var_mean_45[0]
        getitem_97: "f32[1, 160, 1, 1]" = var_mean_45[1];  var_mean_45 = None
        add_229: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 0.001)
        rsqrt_45: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_45: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_97)
        mul_318: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        squeeze_135: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        squeeze_136: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_319: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
        mul_320: "f32[160]" = torch.ops.aten.mul.Tensor(primals_428, 0.9)
        add_230: "f32[160]" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
        squeeze_137: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
        mul_321: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0001081431815724);  squeeze_137 = None
        mul_322: "f32[160]" = torch.ops.aten.mul.Tensor(mul_321, 0.1);  mul_321 = None
        mul_323: "f32[160]" = torch.ops.aten.mul.Tensor(primals_429, 0.9)
        add_231: "f32[160]" = torch.ops.aten.add.Tensor(mul_322, mul_323);  mul_322 = mul_323 = None
        unsqueeze_183: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_137, -1)
        unsqueeze_184: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, -1);  unsqueeze_183 = None
        mul_324: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_318, unsqueeze_184);  mul_318 = unsqueeze_184 = None
        unsqueeze_185: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_138, -1);  primals_138 = None
        unsqueeze_186: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, -1);  unsqueeze_185 = None
        add_232: "f32[32, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_324, unsqueeze_186);  mul_324 = unsqueeze_186 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_45: "f32[32, 160, 17, 17]" = torch.ops.aten.relu.default(add_232);  add_232 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_46: "f32[32, 160, 17, 17]" = torch.ops.aten.convolution.default(relu_45, primals_139, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_233: "i64[]" = torch.ops.aten.add.Tensor(primals_433, 1)
        var_mean_46 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
        getitem_98: "f32[1, 160, 1, 1]" = var_mean_46[0]
        getitem_99: "f32[1, 160, 1, 1]" = var_mean_46[1];  var_mean_46 = None
        add_234: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 0.001)
        rsqrt_46: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_234);  add_234 = None
        sub_46: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_99)
        mul_325: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
        squeeze_138: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
        squeeze_139: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
        mul_326: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
        mul_327: "f32[160]" = torch.ops.aten.mul.Tensor(primals_431, 0.9)
        add_235: "f32[160]" = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
        squeeze_140: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
        mul_328: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0001081431815724);  squeeze_140 = None
        mul_329: "f32[160]" = torch.ops.aten.mul.Tensor(mul_328, 0.1);  mul_328 = None
        mul_330: "f32[160]" = torch.ops.aten.mul.Tensor(primals_432, 0.9)
        add_236: "f32[160]" = torch.ops.aten.add.Tensor(mul_329, mul_330);  mul_329 = mul_330 = None
        unsqueeze_187: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1)
        unsqueeze_188: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, -1);  unsqueeze_187 = None
        mul_331: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_325, unsqueeze_188);  mul_325 = unsqueeze_188 = None
        unsqueeze_189: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1);  primals_141 = None
        unsqueeze_190: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, -1);  unsqueeze_189 = None
        add_237: "f32[32, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_331, unsqueeze_190);  mul_331 = unsqueeze_190 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_46: "f32[32, 160, 17, 17]" = torch.ops.aten.relu.default(add_237);  add_237 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_47: "f32[32, 160, 17, 17]" = torch.ops.aten.convolution.default(relu_46, primals_142, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_238: "i64[]" = torch.ops.aten.add.Tensor(primals_436, 1)
        var_mean_47 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
        getitem_100: "f32[1, 160, 1, 1]" = var_mean_47[0]
        getitem_101: "f32[1, 160, 1, 1]" = var_mean_47[1];  var_mean_47 = None
        add_239: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 0.001)
        rsqrt_47: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        sub_47: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_101)
        mul_332: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
        squeeze_141: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
        squeeze_142: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
        mul_333: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
        mul_334: "f32[160]" = torch.ops.aten.mul.Tensor(primals_434, 0.9)
        add_240: "f32[160]" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
        squeeze_143: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
        mul_335: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0001081431815724);  squeeze_143 = None
        mul_336: "f32[160]" = torch.ops.aten.mul.Tensor(mul_335, 0.1);  mul_335 = None
        mul_337: "f32[160]" = torch.ops.aten.mul.Tensor(primals_435, 0.9)
        add_241: "f32[160]" = torch.ops.aten.add.Tensor(mul_336, mul_337);  mul_336 = mul_337 = None
        unsqueeze_191: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_143, -1)
        unsqueeze_192: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, -1);  unsqueeze_191 = None
        mul_338: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_332, unsqueeze_192);  mul_332 = unsqueeze_192 = None
        unsqueeze_193: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1);  primals_144 = None
        unsqueeze_194: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, -1);  unsqueeze_193 = None
        add_242: "f32[32, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_338, unsqueeze_194);  mul_338 = unsqueeze_194 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_47: "f32[32, 160, 17, 17]" = torch.ops.aten.relu.default(add_242);  add_242 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_48: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_47, primals_145, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_243: "i64[]" = torch.ops.aten.add.Tensor(primals_439, 1)
        var_mean_48 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
        getitem_102: "f32[1, 192, 1, 1]" = var_mean_48[0]
        getitem_103: "f32[1, 192, 1, 1]" = var_mean_48[1];  var_mean_48 = None
        add_244: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 0.001)
        rsqrt_48: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
        sub_48: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_103)
        mul_339: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        squeeze_144: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        squeeze_145: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_340: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
        mul_341: "f32[192]" = torch.ops.aten.mul.Tensor(primals_437, 0.9)
        add_245: "f32[192]" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
        squeeze_146: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
        mul_342: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0001081431815724);  squeeze_146 = None
        mul_343: "f32[192]" = torch.ops.aten.mul.Tensor(mul_342, 0.1);  mul_342 = None
        mul_344: "f32[192]" = torch.ops.aten.mul.Tensor(primals_438, 0.9)
        add_246: "f32[192]" = torch.ops.aten.add.Tensor(mul_343, mul_344);  mul_343 = mul_344 = None
        unsqueeze_195: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_146, -1)
        unsqueeze_196: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, -1);  unsqueeze_195 = None
        mul_345: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_339, unsqueeze_196);  mul_339 = unsqueeze_196 = None
        unsqueeze_197: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_147, -1);  primals_147 = None
        unsqueeze_198: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, -1);  unsqueeze_197 = None
        add_247: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_345, unsqueeze_198);  mul_345 = unsqueeze_198 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_48: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_247);  add_247 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:278 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_4: "f32[32, 768, 17, 17]" = torch.ops.aten.avg_pool2d.default(cat_5, [3, 3], [1, 1], [1, 1])
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_49: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(avg_pool2d_4, primals_148, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_248: "i64[]" = torch.ops.aten.add.Tensor(primals_442, 1)
        var_mean_49 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
        getitem_104: "f32[1, 192, 1, 1]" = var_mean_49[0]
        getitem_105: "f32[1, 192, 1, 1]" = var_mean_49[1];  var_mean_49 = None
        add_249: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 0.001)
        rsqrt_49: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_249);  add_249 = None
        sub_49: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_105)
        mul_346: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
        squeeze_147: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
        squeeze_148: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
        mul_347: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
        mul_348: "f32[192]" = torch.ops.aten.mul.Tensor(primals_440, 0.9)
        add_250: "f32[192]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
        squeeze_149: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
        mul_349: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0001081431815724);  squeeze_149 = None
        mul_350: "f32[192]" = torch.ops.aten.mul.Tensor(mul_349, 0.1);  mul_349 = None
        mul_351: "f32[192]" = torch.ops.aten.mul.Tensor(primals_441, 0.9)
        add_251: "f32[192]" = torch.ops.aten.add.Tensor(mul_350, mul_351);  mul_350 = mul_351 = None
        unsqueeze_199: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_149, -1)
        unsqueeze_200: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, -1);  unsqueeze_199 = None
        mul_352: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_346, unsqueeze_200);  mul_346 = unsqueeze_200 = None
        unsqueeze_201: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_150, -1);  primals_150 = None
        unsqueeze_202: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, -1);  unsqueeze_201 = None
        add_252: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_352, unsqueeze_202);  mul_352 = unsqueeze_202 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_49: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_252);  add_252 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:286 in forward, code: return torch.cat(outputs, 1)
        cat_6: "f32[32, 768, 17, 17]" = torch.ops.aten.cat.default([relu_40, relu_43, relu_48, relu_49], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_50: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(cat_6, primals_151, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_253: "i64[]" = torch.ops.aten.add.Tensor(primals_445, 1)
        var_mean_50 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
        getitem_106: "f32[1, 192, 1, 1]" = var_mean_50[0]
        getitem_107: "f32[1, 192, 1, 1]" = var_mean_50[1];  var_mean_50 = None
        add_254: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 0.001)
        rsqrt_50: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_254);  add_254 = None
        sub_50: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_107)
        mul_353: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
        squeeze_150: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
        squeeze_151: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
        mul_354: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
        mul_355: "f32[192]" = torch.ops.aten.mul.Tensor(primals_443, 0.9)
        add_255: "f32[192]" = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
        squeeze_152: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
        mul_356: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0001081431815724);  squeeze_152 = None
        mul_357: "f32[192]" = torch.ops.aten.mul.Tensor(mul_356, 0.1);  mul_356 = None
        mul_358: "f32[192]" = torch.ops.aten.mul.Tensor(primals_444, 0.9)
        add_256: "f32[192]" = torch.ops.aten.add.Tensor(mul_357, mul_358);  mul_357 = mul_358 = None
        unsqueeze_203: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_152, -1)
        unsqueeze_204: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, -1);  unsqueeze_203 = None
        mul_359: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_353, unsqueeze_204);  mul_353 = unsqueeze_204 = None
        unsqueeze_205: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_153, -1);  primals_153 = None
        unsqueeze_206: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, -1);  unsqueeze_205 = None
        add_257: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_359, unsqueeze_206);  mul_359 = unsqueeze_206 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_50: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_257);  add_257 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_51: "f32[32, 160, 17, 17]" = torch.ops.aten.convolution.default(cat_6, primals_154, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_258: "i64[]" = torch.ops.aten.add.Tensor(primals_448, 1)
        var_mean_51 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
        getitem_108: "f32[1, 160, 1, 1]" = var_mean_51[0]
        getitem_109: "f32[1, 160, 1, 1]" = var_mean_51[1];  var_mean_51 = None
        add_259: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_108, 0.001)
        rsqrt_51: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_259);  add_259 = None
        sub_51: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_109)
        mul_360: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
        squeeze_153: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
        squeeze_154: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_361: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
        mul_362: "f32[160]" = torch.ops.aten.mul.Tensor(primals_446, 0.9)
        add_260: "f32[160]" = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
        squeeze_155: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
        mul_363: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0001081431815724);  squeeze_155 = None
        mul_364: "f32[160]" = torch.ops.aten.mul.Tensor(mul_363, 0.1);  mul_363 = None
        mul_365: "f32[160]" = torch.ops.aten.mul.Tensor(primals_447, 0.9)
        add_261: "f32[160]" = torch.ops.aten.add.Tensor(mul_364, mul_365);  mul_364 = mul_365 = None
        unsqueeze_207: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_155, -1)
        unsqueeze_208: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, -1);  unsqueeze_207 = None
        mul_366: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_360, unsqueeze_208);  mul_360 = unsqueeze_208 = None
        unsqueeze_209: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1);  primals_156 = None
        unsqueeze_210: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, -1);  unsqueeze_209 = None
        add_262: "f32[32, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_366, unsqueeze_210);  mul_366 = unsqueeze_210 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_51: "f32[32, 160, 17, 17]" = torch.ops.aten.relu.default(add_262);  add_262 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_52: "f32[32, 160, 17, 17]" = torch.ops.aten.convolution.default(relu_51, primals_157, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_263: "i64[]" = torch.ops.aten.add.Tensor(primals_451, 1)
        var_mean_52 = torch.ops.aten.var_mean.correction(convolution_52, [0, 2, 3], correction = 0, keepdim = True)
        getitem_110: "f32[1, 160, 1, 1]" = var_mean_52[0]
        getitem_111: "f32[1, 160, 1, 1]" = var_mean_52[1];  var_mean_52 = None
        add_264: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 0.001)
        rsqrt_52: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_264);  add_264 = None
        sub_52: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_52, getitem_111)
        mul_367: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
        squeeze_156: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
        squeeze_157: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
        mul_368: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
        mul_369: "f32[160]" = torch.ops.aten.mul.Tensor(primals_449, 0.9)
        add_265: "f32[160]" = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
        squeeze_158: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
        mul_370: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0001081431815724);  squeeze_158 = None
        mul_371: "f32[160]" = torch.ops.aten.mul.Tensor(mul_370, 0.1);  mul_370 = None
        mul_372: "f32[160]" = torch.ops.aten.mul.Tensor(primals_450, 0.9)
        add_266: "f32[160]" = torch.ops.aten.add.Tensor(mul_371, mul_372);  mul_371 = mul_372 = None
        unsqueeze_211: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_158, -1)
        unsqueeze_212: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, -1);  unsqueeze_211 = None
        mul_373: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_367, unsqueeze_212);  mul_367 = unsqueeze_212 = None
        unsqueeze_213: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_159, -1);  primals_159 = None
        unsqueeze_214: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, -1);  unsqueeze_213 = None
        add_267: "f32[32, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_373, unsqueeze_214);  mul_373 = unsqueeze_214 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_52: "f32[32, 160, 17, 17]" = torch.ops.aten.relu.default(add_267);  add_267 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_53: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_52, primals_160, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_268: "i64[]" = torch.ops.aten.add.Tensor(primals_454, 1)
        var_mean_53 = torch.ops.aten.var_mean.correction(convolution_53, [0, 2, 3], correction = 0, keepdim = True)
        getitem_112: "f32[1, 192, 1, 1]" = var_mean_53[0]
        getitem_113: "f32[1, 192, 1, 1]" = var_mean_53[1];  var_mean_53 = None
        add_269: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 0.001)
        rsqrt_53: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_269);  add_269 = None
        sub_53: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_53, getitem_113)
        mul_374: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
        squeeze_159: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
        squeeze_160: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
        mul_375: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
        mul_376: "f32[192]" = torch.ops.aten.mul.Tensor(primals_452, 0.9)
        add_270: "f32[192]" = torch.ops.aten.add.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
        squeeze_161: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
        mul_377: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0001081431815724);  squeeze_161 = None
        mul_378: "f32[192]" = torch.ops.aten.mul.Tensor(mul_377, 0.1);  mul_377 = None
        mul_379: "f32[192]" = torch.ops.aten.mul.Tensor(primals_453, 0.9)
        add_271: "f32[192]" = torch.ops.aten.add.Tensor(mul_378, mul_379);  mul_378 = mul_379 = None
        unsqueeze_215: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_161, -1)
        unsqueeze_216: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, -1);  unsqueeze_215 = None
        mul_380: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_374, unsqueeze_216);  mul_374 = unsqueeze_216 = None
        unsqueeze_217: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_162, -1);  primals_162 = None
        unsqueeze_218: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, -1);  unsqueeze_217 = None
        add_272: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_380, unsqueeze_218);  mul_380 = unsqueeze_218 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_53: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_272);  add_272 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_54: "f32[32, 160, 17, 17]" = torch.ops.aten.convolution.default(cat_6, primals_163, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_273: "i64[]" = torch.ops.aten.add.Tensor(primals_457, 1)
        var_mean_54 = torch.ops.aten.var_mean.correction(convolution_54, [0, 2, 3], correction = 0, keepdim = True)
        getitem_114: "f32[1, 160, 1, 1]" = var_mean_54[0]
        getitem_115: "f32[1, 160, 1, 1]" = var_mean_54[1];  var_mean_54 = None
        add_274: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_114, 0.001)
        rsqrt_54: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_274);  add_274 = None
        sub_54: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_115)
        mul_381: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
        squeeze_162: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
        squeeze_163: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
        mul_382: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
        mul_383: "f32[160]" = torch.ops.aten.mul.Tensor(primals_455, 0.9)
        add_275: "f32[160]" = torch.ops.aten.add.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
        squeeze_164: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
        mul_384: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0001081431815724);  squeeze_164 = None
        mul_385: "f32[160]" = torch.ops.aten.mul.Tensor(mul_384, 0.1);  mul_384 = None
        mul_386: "f32[160]" = torch.ops.aten.mul.Tensor(primals_456, 0.9)
        add_276: "f32[160]" = torch.ops.aten.add.Tensor(mul_385, mul_386);  mul_385 = mul_386 = None
        unsqueeze_219: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_164, -1)
        unsqueeze_220: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, -1);  unsqueeze_219 = None
        mul_387: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_381, unsqueeze_220);  mul_381 = unsqueeze_220 = None
        unsqueeze_221: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_165, -1);  primals_165 = None
        unsqueeze_222: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, -1);  unsqueeze_221 = None
        add_277: "f32[32, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_387, unsqueeze_222);  mul_387 = unsqueeze_222 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_54: "f32[32, 160, 17, 17]" = torch.ops.aten.relu.default(add_277);  add_277 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_55: "f32[32, 160, 17, 17]" = torch.ops.aten.convolution.default(relu_54, primals_166, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_278: "i64[]" = torch.ops.aten.add.Tensor(primals_460, 1)
        var_mean_55 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
        getitem_116: "f32[1, 160, 1, 1]" = var_mean_55[0]
        getitem_117: "f32[1, 160, 1, 1]" = var_mean_55[1];  var_mean_55 = None
        add_279: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_116, 0.001)
        rsqrt_55: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_279);  add_279 = None
        sub_55: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_117)
        mul_388: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
        squeeze_165: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
        squeeze_166: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
        mul_389: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
        mul_390: "f32[160]" = torch.ops.aten.mul.Tensor(primals_458, 0.9)
        add_280: "f32[160]" = torch.ops.aten.add.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
        squeeze_167: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
        mul_391: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0001081431815724);  squeeze_167 = None
        mul_392: "f32[160]" = torch.ops.aten.mul.Tensor(mul_391, 0.1);  mul_391 = None
        mul_393: "f32[160]" = torch.ops.aten.mul.Tensor(primals_459, 0.9)
        add_281: "f32[160]" = torch.ops.aten.add.Tensor(mul_392, mul_393);  mul_392 = mul_393 = None
        unsqueeze_223: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_167, -1)
        unsqueeze_224: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, -1);  unsqueeze_223 = None
        mul_394: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_388, unsqueeze_224);  mul_388 = unsqueeze_224 = None
        unsqueeze_225: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_168, -1);  primals_168 = None
        unsqueeze_226: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, -1);  unsqueeze_225 = None
        add_282: "f32[32, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_394, unsqueeze_226);  mul_394 = unsqueeze_226 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_55: "f32[32, 160, 17, 17]" = torch.ops.aten.relu.default(add_282);  add_282 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_56: "f32[32, 160, 17, 17]" = torch.ops.aten.convolution.default(relu_55, primals_169, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_283: "i64[]" = torch.ops.aten.add.Tensor(primals_463, 1)
        var_mean_56 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
        getitem_118: "f32[1, 160, 1, 1]" = var_mean_56[0]
        getitem_119: "f32[1, 160, 1, 1]" = var_mean_56[1];  var_mean_56 = None
        add_284: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 0.001)
        rsqrt_56: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_284);  add_284 = None
        sub_56: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_56, getitem_119)
        mul_395: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
        squeeze_168: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
        squeeze_169: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
        mul_396: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
        mul_397: "f32[160]" = torch.ops.aten.mul.Tensor(primals_461, 0.9)
        add_285: "f32[160]" = torch.ops.aten.add.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
        squeeze_170: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
        mul_398: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0001081431815724);  squeeze_170 = None
        mul_399: "f32[160]" = torch.ops.aten.mul.Tensor(mul_398, 0.1);  mul_398 = None
        mul_400: "f32[160]" = torch.ops.aten.mul.Tensor(primals_462, 0.9)
        add_286: "f32[160]" = torch.ops.aten.add.Tensor(mul_399, mul_400);  mul_399 = mul_400 = None
        unsqueeze_227: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_170, -1)
        unsqueeze_228: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, -1);  unsqueeze_227 = None
        mul_401: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_395, unsqueeze_228);  mul_395 = unsqueeze_228 = None
        unsqueeze_229: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_171, -1);  primals_171 = None
        unsqueeze_230: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, -1);  unsqueeze_229 = None
        add_287: "f32[32, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_401, unsqueeze_230);  mul_401 = unsqueeze_230 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_56: "f32[32, 160, 17, 17]" = torch.ops.aten.relu.default(add_287);  add_287 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_57: "f32[32, 160, 17, 17]" = torch.ops.aten.convolution.default(relu_56, primals_172, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_288: "i64[]" = torch.ops.aten.add.Tensor(primals_466, 1)
        var_mean_57 = torch.ops.aten.var_mean.correction(convolution_57, [0, 2, 3], correction = 0, keepdim = True)
        getitem_120: "f32[1, 160, 1, 1]" = var_mean_57[0]
        getitem_121: "f32[1, 160, 1, 1]" = var_mean_57[1];  var_mean_57 = None
        add_289: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 0.001)
        rsqrt_57: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_289);  add_289 = None
        sub_57: "f32[32, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_57, getitem_121)
        mul_402: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
        squeeze_171: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
        squeeze_172: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
        mul_403: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
        mul_404: "f32[160]" = torch.ops.aten.mul.Tensor(primals_464, 0.9)
        add_290: "f32[160]" = torch.ops.aten.add.Tensor(mul_403, mul_404);  mul_403 = mul_404 = None
        squeeze_173: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
        mul_405: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0001081431815724);  squeeze_173 = None
        mul_406: "f32[160]" = torch.ops.aten.mul.Tensor(mul_405, 0.1);  mul_405 = None
        mul_407: "f32[160]" = torch.ops.aten.mul.Tensor(primals_465, 0.9)
        add_291: "f32[160]" = torch.ops.aten.add.Tensor(mul_406, mul_407);  mul_406 = mul_407 = None
        unsqueeze_231: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_173, -1)
        unsqueeze_232: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, -1);  unsqueeze_231 = None
        mul_408: "f32[32, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_402, unsqueeze_232);  mul_402 = unsqueeze_232 = None
        unsqueeze_233: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_174, -1);  primals_174 = None
        unsqueeze_234: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, -1);  unsqueeze_233 = None
        add_292: "f32[32, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_408, unsqueeze_234);  mul_408 = unsqueeze_234 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_57: "f32[32, 160, 17, 17]" = torch.ops.aten.relu.default(add_292);  add_292 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_58: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_57, primals_175, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_293: "i64[]" = torch.ops.aten.add.Tensor(primals_469, 1)
        var_mean_58 = torch.ops.aten.var_mean.correction(convolution_58, [0, 2, 3], correction = 0, keepdim = True)
        getitem_122: "f32[1, 192, 1, 1]" = var_mean_58[0]
        getitem_123: "f32[1, 192, 1, 1]" = var_mean_58[1];  var_mean_58 = None
        add_294: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_122, 0.001)
        rsqrt_58: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_294);  add_294 = None
        sub_58: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_58, getitem_123)
        mul_409: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
        squeeze_174: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
        squeeze_175: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
        mul_410: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1)
        mul_411: "f32[192]" = torch.ops.aten.mul.Tensor(primals_467, 0.9)
        add_295: "f32[192]" = torch.ops.aten.add.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
        squeeze_176: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
        mul_412: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_176, 1.0001081431815724);  squeeze_176 = None
        mul_413: "f32[192]" = torch.ops.aten.mul.Tensor(mul_412, 0.1);  mul_412 = None
        mul_414: "f32[192]" = torch.ops.aten.mul.Tensor(primals_468, 0.9)
        add_296: "f32[192]" = torch.ops.aten.add.Tensor(mul_413, mul_414);  mul_413 = mul_414 = None
        unsqueeze_235: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_176, -1)
        unsqueeze_236: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, -1);  unsqueeze_235 = None
        mul_415: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_409, unsqueeze_236);  mul_409 = unsqueeze_236 = None
        unsqueeze_237: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_177, -1);  primals_177 = None
        unsqueeze_238: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, -1);  unsqueeze_237 = None
        add_297: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_415, unsqueeze_238);  mul_415 = unsqueeze_238 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_58: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_297);  add_297 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:278 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_5: "f32[32, 768, 17, 17]" = torch.ops.aten.avg_pool2d.default(cat_6, [3, 3], [1, 1], [1, 1])
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_59: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(avg_pool2d_5, primals_178, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_298: "i64[]" = torch.ops.aten.add.Tensor(primals_472, 1)
        var_mean_59 = torch.ops.aten.var_mean.correction(convolution_59, [0, 2, 3], correction = 0, keepdim = True)
        getitem_124: "f32[1, 192, 1, 1]" = var_mean_59[0]
        getitem_125: "f32[1, 192, 1, 1]" = var_mean_59[1];  var_mean_59 = None
        add_299: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 0.001)
        rsqrt_59: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_299);  add_299 = None
        sub_59: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_59, getitem_125)
        mul_416: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
        squeeze_177: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
        squeeze_178: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
        mul_417: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1)
        mul_418: "f32[192]" = torch.ops.aten.mul.Tensor(primals_470, 0.9)
        add_300: "f32[192]" = torch.ops.aten.add.Tensor(mul_417, mul_418);  mul_417 = mul_418 = None
        squeeze_179: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
        mul_419: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_179, 1.0001081431815724);  squeeze_179 = None
        mul_420: "f32[192]" = torch.ops.aten.mul.Tensor(mul_419, 0.1);  mul_419 = None
        mul_421: "f32[192]" = torch.ops.aten.mul.Tensor(primals_471, 0.9)
        add_301: "f32[192]" = torch.ops.aten.add.Tensor(mul_420, mul_421);  mul_420 = mul_421 = None
        unsqueeze_239: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_179, -1)
        unsqueeze_240: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, -1);  unsqueeze_239 = None
        mul_422: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_416, unsqueeze_240);  mul_416 = unsqueeze_240 = None
        unsqueeze_241: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_180, -1);  primals_180 = None
        unsqueeze_242: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, -1);  unsqueeze_241 = None
        add_302: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_422, unsqueeze_242);  mul_422 = unsqueeze_242 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_59: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_302);  add_302 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:286 in forward, code: return torch.cat(outputs, 1)
        cat_7: "f32[32, 768, 17, 17]" = torch.ops.aten.cat.default([relu_50, relu_53, relu_58, relu_59], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_60: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(cat_7, primals_181, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_303: "i64[]" = torch.ops.aten.add.Tensor(primals_475, 1)
        var_mean_60 = torch.ops.aten.var_mean.correction(convolution_60, [0, 2, 3], correction = 0, keepdim = True)
        getitem_126: "f32[1, 192, 1, 1]" = var_mean_60[0]
        getitem_127: "f32[1, 192, 1, 1]" = var_mean_60[1];  var_mean_60 = None
        add_304: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_126, 0.001)
        rsqrt_60: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_304);  add_304 = None
        sub_60: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_127)
        mul_423: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
        squeeze_180: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
        squeeze_181: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
        mul_424: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1)
        mul_425: "f32[192]" = torch.ops.aten.mul.Tensor(primals_473, 0.9)
        add_305: "f32[192]" = torch.ops.aten.add.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None
        squeeze_182: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
        mul_426: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_182, 1.0001081431815724);  squeeze_182 = None
        mul_427: "f32[192]" = torch.ops.aten.mul.Tensor(mul_426, 0.1);  mul_426 = None
        mul_428: "f32[192]" = torch.ops.aten.mul.Tensor(primals_474, 0.9)
        add_306: "f32[192]" = torch.ops.aten.add.Tensor(mul_427, mul_428);  mul_427 = mul_428 = None
        unsqueeze_243: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_182, -1)
        unsqueeze_244: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, -1);  unsqueeze_243 = None
        mul_429: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_423, unsqueeze_244);  mul_423 = unsqueeze_244 = None
        unsqueeze_245: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_183, -1);  primals_183 = None
        unsqueeze_246: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, -1);  unsqueeze_245 = None
        add_307: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_429, unsqueeze_246);  mul_429 = unsqueeze_246 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_60: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_307);  add_307 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_61: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(cat_7, primals_184, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_308: "i64[]" = torch.ops.aten.add.Tensor(primals_478, 1)
        var_mean_61 = torch.ops.aten.var_mean.correction(convolution_61, [0, 2, 3], correction = 0, keepdim = True)
        getitem_128: "f32[1, 192, 1, 1]" = var_mean_61[0]
        getitem_129: "f32[1, 192, 1, 1]" = var_mean_61[1];  var_mean_61 = None
        add_309: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_128, 0.001)
        rsqrt_61: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_309);  add_309 = None
        sub_61: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_61, getitem_129)
        mul_430: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = None
        squeeze_183: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
        squeeze_184: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
        mul_431: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1)
        mul_432: "f32[192]" = torch.ops.aten.mul.Tensor(primals_476, 0.9)
        add_310: "f32[192]" = torch.ops.aten.add.Tensor(mul_431, mul_432);  mul_431 = mul_432 = None
        squeeze_185: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
        mul_433: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_185, 1.0001081431815724);  squeeze_185 = None
        mul_434: "f32[192]" = torch.ops.aten.mul.Tensor(mul_433, 0.1);  mul_433 = None
        mul_435: "f32[192]" = torch.ops.aten.mul.Tensor(primals_477, 0.9)
        add_311: "f32[192]" = torch.ops.aten.add.Tensor(mul_434, mul_435);  mul_434 = mul_435 = None
        unsqueeze_247: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_185, -1)
        unsqueeze_248: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, -1);  unsqueeze_247 = None
        mul_436: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_430, unsqueeze_248);  mul_430 = unsqueeze_248 = None
        unsqueeze_249: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_186, -1);  primals_186 = None
        unsqueeze_250: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, -1);  unsqueeze_249 = None
        add_312: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_436, unsqueeze_250);  mul_436 = unsqueeze_250 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_61: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_312);  add_312 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_62: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_61, primals_187, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_313: "i64[]" = torch.ops.aten.add.Tensor(primals_481, 1)
        var_mean_62 = torch.ops.aten.var_mean.correction(convolution_62, [0, 2, 3], correction = 0, keepdim = True)
        getitem_130: "f32[1, 192, 1, 1]" = var_mean_62[0]
        getitem_131: "f32[1, 192, 1, 1]" = var_mean_62[1];  var_mean_62 = None
        add_314: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_130, 0.001)
        rsqrt_62: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_314);  add_314 = None
        sub_62: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_62, getitem_131)
        mul_437: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = None
        squeeze_186: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_131, [0, 2, 3]);  getitem_131 = None
        squeeze_187: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_62, [0, 2, 3]);  rsqrt_62 = None
        mul_438: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_186, 0.1)
        mul_439: "f32[192]" = torch.ops.aten.mul.Tensor(primals_479, 0.9)
        add_315: "f32[192]" = torch.ops.aten.add.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
        squeeze_188: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_130, [0, 2, 3]);  getitem_130 = None
        mul_440: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_188, 1.0001081431815724);  squeeze_188 = None
        mul_441: "f32[192]" = torch.ops.aten.mul.Tensor(mul_440, 0.1);  mul_440 = None
        mul_442: "f32[192]" = torch.ops.aten.mul.Tensor(primals_480, 0.9)
        add_316: "f32[192]" = torch.ops.aten.add.Tensor(mul_441, mul_442);  mul_441 = mul_442 = None
        unsqueeze_251: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_188, -1)
        unsqueeze_252: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, -1);  unsqueeze_251 = None
        mul_443: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_437, unsqueeze_252);  mul_437 = unsqueeze_252 = None
        unsqueeze_253: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_189, -1);  primals_189 = None
        unsqueeze_254: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, -1);  unsqueeze_253 = None
        add_317: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_443, unsqueeze_254);  mul_443 = unsqueeze_254 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_62: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_317);  add_317 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_63: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_62, primals_190, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_318: "i64[]" = torch.ops.aten.add.Tensor(primals_484, 1)
        var_mean_63 = torch.ops.aten.var_mean.correction(convolution_63, [0, 2, 3], correction = 0, keepdim = True)
        getitem_132: "f32[1, 192, 1, 1]" = var_mean_63[0]
        getitem_133: "f32[1, 192, 1, 1]" = var_mean_63[1];  var_mean_63 = None
        add_319: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_132, 0.001)
        rsqrt_63: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_319);  add_319 = None
        sub_63: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_63, getitem_133)
        mul_444: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = None
        squeeze_189: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_133, [0, 2, 3]);  getitem_133 = None
        squeeze_190: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
        mul_445: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_189, 0.1)
        mul_446: "f32[192]" = torch.ops.aten.mul.Tensor(primals_482, 0.9)
        add_320: "f32[192]" = torch.ops.aten.add.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None
        squeeze_191: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_132, [0, 2, 3]);  getitem_132 = None
        mul_447: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_191, 1.0001081431815724);  squeeze_191 = None
        mul_448: "f32[192]" = torch.ops.aten.mul.Tensor(mul_447, 0.1);  mul_447 = None
        mul_449: "f32[192]" = torch.ops.aten.mul.Tensor(primals_483, 0.9)
        add_321: "f32[192]" = torch.ops.aten.add.Tensor(mul_448, mul_449);  mul_448 = mul_449 = None
        unsqueeze_255: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_191, -1)
        unsqueeze_256: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, -1);  unsqueeze_255 = None
        mul_450: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_444, unsqueeze_256);  mul_444 = unsqueeze_256 = None
        unsqueeze_257: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_192, -1);  primals_192 = None
        unsqueeze_258: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, -1);  unsqueeze_257 = None
        add_322: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_450, unsqueeze_258);  mul_450 = unsqueeze_258 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_63: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_322);  add_322 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_64: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(cat_7, primals_193, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_323: "i64[]" = torch.ops.aten.add.Tensor(primals_487, 1)
        var_mean_64 = torch.ops.aten.var_mean.correction(convolution_64, [0, 2, 3], correction = 0, keepdim = True)
        getitem_134: "f32[1, 192, 1, 1]" = var_mean_64[0]
        getitem_135: "f32[1, 192, 1, 1]" = var_mean_64[1];  var_mean_64 = None
        add_324: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_134, 0.001)
        rsqrt_64: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_324);  add_324 = None
        sub_64: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_64, getitem_135)
        mul_451: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = None
        squeeze_192: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_135, [0, 2, 3]);  getitem_135 = None
        squeeze_193: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_64, [0, 2, 3]);  rsqrt_64 = None
        mul_452: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_192, 0.1)
        mul_453: "f32[192]" = torch.ops.aten.mul.Tensor(primals_485, 0.9)
        add_325: "f32[192]" = torch.ops.aten.add.Tensor(mul_452, mul_453);  mul_452 = mul_453 = None
        squeeze_194: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_134, [0, 2, 3]);  getitem_134 = None
        mul_454: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_194, 1.0001081431815724);  squeeze_194 = None
        mul_455: "f32[192]" = torch.ops.aten.mul.Tensor(mul_454, 0.1);  mul_454 = None
        mul_456: "f32[192]" = torch.ops.aten.mul.Tensor(primals_486, 0.9)
        add_326: "f32[192]" = torch.ops.aten.add.Tensor(mul_455, mul_456);  mul_455 = mul_456 = None
        unsqueeze_259: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_194, -1)
        unsqueeze_260: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, -1);  unsqueeze_259 = None
        mul_457: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_451, unsqueeze_260);  mul_451 = unsqueeze_260 = None
        unsqueeze_261: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_195, -1);  primals_195 = None
        unsqueeze_262: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, -1);  unsqueeze_261 = None
        add_327: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_457, unsqueeze_262);  mul_457 = unsqueeze_262 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_64: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_327);  add_327 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_65: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_64, primals_196, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_328: "i64[]" = torch.ops.aten.add.Tensor(primals_490, 1)
        var_mean_65 = torch.ops.aten.var_mean.correction(convolution_65, [0, 2, 3], correction = 0, keepdim = True)
        getitem_136: "f32[1, 192, 1, 1]" = var_mean_65[0]
        getitem_137: "f32[1, 192, 1, 1]" = var_mean_65[1];  var_mean_65 = None
        add_329: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_136, 0.001)
        rsqrt_65: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_329);  add_329 = None
        sub_65: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_65, getitem_137)
        mul_458: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = None
        squeeze_195: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3]);  getitem_137 = None
        squeeze_196: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_65, [0, 2, 3]);  rsqrt_65 = None
        mul_459: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_195, 0.1)
        mul_460: "f32[192]" = torch.ops.aten.mul.Tensor(primals_488, 0.9)
        add_330: "f32[192]" = torch.ops.aten.add.Tensor(mul_459, mul_460);  mul_459 = mul_460 = None
        squeeze_197: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_136, [0, 2, 3]);  getitem_136 = None
        mul_461: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_197, 1.0001081431815724);  squeeze_197 = None
        mul_462: "f32[192]" = torch.ops.aten.mul.Tensor(mul_461, 0.1);  mul_461 = None
        mul_463: "f32[192]" = torch.ops.aten.mul.Tensor(primals_489, 0.9)
        add_331: "f32[192]" = torch.ops.aten.add.Tensor(mul_462, mul_463);  mul_462 = mul_463 = None
        unsqueeze_263: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_197, -1)
        unsqueeze_264: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, -1);  unsqueeze_263 = None
        mul_464: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_458, unsqueeze_264);  mul_458 = unsqueeze_264 = None
        unsqueeze_265: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_198, -1);  primals_198 = None
        unsqueeze_266: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, -1);  unsqueeze_265 = None
        add_332: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_464, unsqueeze_266);  mul_464 = unsqueeze_266 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_65: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_332);  add_332 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_66: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_65, primals_199, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_333: "i64[]" = torch.ops.aten.add.Tensor(primals_493, 1)
        var_mean_66 = torch.ops.aten.var_mean.correction(convolution_66, [0, 2, 3], correction = 0, keepdim = True)
        getitem_138: "f32[1, 192, 1, 1]" = var_mean_66[0]
        getitem_139: "f32[1, 192, 1, 1]" = var_mean_66[1];  var_mean_66 = None
        add_334: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_138, 0.001)
        rsqrt_66: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_334);  add_334 = None
        sub_66: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_66, getitem_139)
        mul_465: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = None
        squeeze_198: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3]);  getitem_139 = None
        squeeze_199: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_66, [0, 2, 3]);  rsqrt_66 = None
        mul_466: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_198, 0.1)
        mul_467: "f32[192]" = torch.ops.aten.mul.Tensor(primals_491, 0.9)
        add_335: "f32[192]" = torch.ops.aten.add.Tensor(mul_466, mul_467);  mul_466 = mul_467 = None
        squeeze_200: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_138, [0, 2, 3]);  getitem_138 = None
        mul_468: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_200, 1.0001081431815724);  squeeze_200 = None
        mul_469: "f32[192]" = torch.ops.aten.mul.Tensor(mul_468, 0.1);  mul_468 = None
        mul_470: "f32[192]" = torch.ops.aten.mul.Tensor(primals_492, 0.9)
        add_336: "f32[192]" = torch.ops.aten.add.Tensor(mul_469, mul_470);  mul_469 = mul_470 = None
        unsqueeze_267: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_200, -1)
        unsqueeze_268: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, -1);  unsqueeze_267 = None
        mul_471: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_465, unsqueeze_268);  mul_465 = unsqueeze_268 = None
        unsqueeze_269: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_201, -1);  primals_201 = None
        unsqueeze_270: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, -1);  unsqueeze_269 = None
        add_337: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_471, unsqueeze_270);  mul_471 = unsqueeze_270 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_66: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_337);  add_337 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_67: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_66, primals_202, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_338: "i64[]" = torch.ops.aten.add.Tensor(primals_496, 1)
        var_mean_67 = torch.ops.aten.var_mean.correction(convolution_67, [0, 2, 3], correction = 0, keepdim = True)
        getitem_140: "f32[1, 192, 1, 1]" = var_mean_67[0]
        getitem_141: "f32[1, 192, 1, 1]" = var_mean_67[1];  var_mean_67 = None
        add_339: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_140, 0.001)
        rsqrt_67: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_339);  add_339 = None
        sub_67: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_67, getitem_141)
        mul_472: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = None
        squeeze_201: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_141, [0, 2, 3]);  getitem_141 = None
        squeeze_202: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_67, [0, 2, 3]);  rsqrt_67 = None
        mul_473: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_201, 0.1)
        mul_474: "f32[192]" = torch.ops.aten.mul.Tensor(primals_494, 0.9)
        add_340: "f32[192]" = torch.ops.aten.add.Tensor(mul_473, mul_474);  mul_473 = mul_474 = None
        squeeze_203: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_140, [0, 2, 3]);  getitem_140 = None
        mul_475: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_203, 1.0001081431815724);  squeeze_203 = None
        mul_476: "f32[192]" = torch.ops.aten.mul.Tensor(mul_475, 0.1);  mul_475 = None
        mul_477: "f32[192]" = torch.ops.aten.mul.Tensor(primals_495, 0.9)
        add_341: "f32[192]" = torch.ops.aten.add.Tensor(mul_476, mul_477);  mul_476 = mul_477 = None
        unsqueeze_271: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_203, -1)
        unsqueeze_272: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, -1);  unsqueeze_271 = None
        mul_478: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_472, unsqueeze_272);  mul_472 = unsqueeze_272 = None
        unsqueeze_273: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_204, -1);  primals_204 = None
        unsqueeze_274: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, -1);  unsqueeze_273 = None
        add_342: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_478, unsqueeze_274);  mul_478 = unsqueeze_274 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_67: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_342);  add_342 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_68: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_67, primals_205, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_343: "i64[]" = torch.ops.aten.add.Tensor(primals_499, 1)
        var_mean_68 = torch.ops.aten.var_mean.correction(convolution_68, [0, 2, 3], correction = 0, keepdim = True)
        getitem_142: "f32[1, 192, 1, 1]" = var_mean_68[0]
        getitem_143: "f32[1, 192, 1, 1]" = var_mean_68[1];  var_mean_68 = None
        add_344: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_142, 0.001)
        rsqrt_68: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_344);  add_344 = None
        sub_68: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_68, getitem_143)
        mul_479: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = None
        squeeze_204: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_143, [0, 2, 3]);  getitem_143 = None
        squeeze_205: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_68, [0, 2, 3]);  rsqrt_68 = None
        mul_480: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_204, 0.1)
        mul_481: "f32[192]" = torch.ops.aten.mul.Tensor(primals_497, 0.9)
        add_345: "f32[192]" = torch.ops.aten.add.Tensor(mul_480, mul_481);  mul_480 = mul_481 = None
        squeeze_206: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_142, [0, 2, 3]);  getitem_142 = None
        mul_482: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_206, 1.0001081431815724);  squeeze_206 = None
        mul_483: "f32[192]" = torch.ops.aten.mul.Tensor(mul_482, 0.1);  mul_482 = None
        mul_484: "f32[192]" = torch.ops.aten.mul.Tensor(primals_498, 0.9)
        add_346: "f32[192]" = torch.ops.aten.add.Tensor(mul_483, mul_484);  mul_483 = mul_484 = None
        unsqueeze_275: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_206, -1)
        unsqueeze_276: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, -1);  unsqueeze_275 = None
        mul_485: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_479, unsqueeze_276);  mul_479 = unsqueeze_276 = None
        unsqueeze_277: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_207, -1);  primals_207 = None
        unsqueeze_278: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, -1);  unsqueeze_277 = None
        add_347: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_485, unsqueeze_278);  mul_485 = unsqueeze_278 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_68: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_347);  add_347 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:278 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_6: "f32[32, 768, 17, 17]" = torch.ops.aten.avg_pool2d.default(cat_7, [3, 3], [1, 1], [1, 1])
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_69: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(avg_pool2d_6, primals_208, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_348: "i64[]" = torch.ops.aten.add.Tensor(primals_502, 1)
        var_mean_69 = torch.ops.aten.var_mean.correction(convolution_69, [0, 2, 3], correction = 0, keepdim = True)
        getitem_144: "f32[1, 192, 1, 1]" = var_mean_69[0]
        getitem_145: "f32[1, 192, 1, 1]" = var_mean_69[1];  var_mean_69 = None
        add_349: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_144, 0.001)
        rsqrt_69: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_349);  add_349 = None
        sub_69: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_69, getitem_145)
        mul_486: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = None
        squeeze_207: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
        squeeze_208: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_69, [0, 2, 3]);  rsqrt_69 = None
        mul_487: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_207, 0.1)
        mul_488: "f32[192]" = torch.ops.aten.mul.Tensor(primals_500, 0.9)
        add_350: "f32[192]" = torch.ops.aten.add.Tensor(mul_487, mul_488);  mul_487 = mul_488 = None
        squeeze_209: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_144, [0, 2, 3]);  getitem_144 = None
        mul_489: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_209, 1.0001081431815724);  squeeze_209 = None
        mul_490: "f32[192]" = torch.ops.aten.mul.Tensor(mul_489, 0.1);  mul_489 = None
        mul_491: "f32[192]" = torch.ops.aten.mul.Tensor(primals_501, 0.9)
        add_351: "f32[192]" = torch.ops.aten.add.Tensor(mul_490, mul_491);  mul_490 = mul_491 = None
        unsqueeze_279: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_209, -1)
        unsqueeze_280: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, -1);  unsqueeze_279 = None
        mul_492: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_486, unsqueeze_280);  mul_486 = unsqueeze_280 = None
        unsqueeze_281: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_210, -1);  primals_210 = None
        unsqueeze_282: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, -1);  unsqueeze_281 = None
        add_352: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_492, unsqueeze_282);  mul_492 = unsqueeze_282 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_69: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_352);  add_352 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:286 in forward, code: return torch.cat(outputs, 1)
        cat_8: "f32[32, 768, 17, 17]" = torch.ops.aten.cat.default([relu_60, relu_63, relu_68, relu_69], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:382 in forward, code: x = F.avg_pool2d(x, kernel_size=5, stride=3)
        avg_pool2d_7: "f32[32, 768, 5, 5]" = torch.ops.aten.avg_pool2d.default(cat_8, [5, 5], [3, 3])
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_70: "f32[32, 128, 5, 5]" = torch.ops.aten.convolution.default(avg_pool2d_7, primals_211, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_353: "i64[]" = torch.ops.aten.add.Tensor(primals_505, 1)
        var_mean_70 = torch.ops.aten.var_mean.correction(convolution_70, [0, 2, 3], correction = 0, keepdim = True)
        getitem_146: "f32[1, 128, 1, 1]" = var_mean_70[0]
        getitem_147: "f32[1, 128, 1, 1]" = var_mean_70[1];  var_mean_70 = None
        add_354: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_146, 0.001)
        rsqrt_70: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_354);  add_354 = None
        sub_70: "f32[32, 128, 5, 5]" = torch.ops.aten.sub.Tensor(convolution_70, getitem_147)
        mul_493: "f32[32, 128, 5, 5]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = None
        squeeze_210: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_147, [0, 2, 3]);  getitem_147 = None
        squeeze_211: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_70, [0, 2, 3]);  rsqrt_70 = None
        mul_494: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_210, 0.1)
        mul_495: "f32[128]" = torch.ops.aten.mul.Tensor(primals_503, 0.9)
        add_355: "f32[128]" = torch.ops.aten.add.Tensor(mul_494, mul_495);  mul_494 = mul_495 = None
        squeeze_212: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_146, [0, 2, 3]);  getitem_146 = None
        mul_496: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_212, 1.0012515644555695);  squeeze_212 = None
        mul_497: "f32[128]" = torch.ops.aten.mul.Tensor(mul_496, 0.1);  mul_496 = None
        mul_498: "f32[128]" = torch.ops.aten.mul.Tensor(primals_504, 0.9)
        add_356: "f32[128]" = torch.ops.aten.add.Tensor(mul_497, mul_498);  mul_497 = mul_498 = None
        unsqueeze_283: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_212, -1)
        unsqueeze_284: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, -1);  unsqueeze_283 = None
        mul_499: "f32[32, 128, 5, 5]" = torch.ops.aten.mul.Tensor(mul_493, unsqueeze_284);  mul_493 = unsqueeze_284 = None
        unsqueeze_285: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_213, -1);  primals_213 = None
        unsqueeze_286: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, -1);  unsqueeze_285 = None
        add_357: "f32[32, 128, 5, 5]" = torch.ops.aten.add.Tensor(mul_499, unsqueeze_286);  mul_499 = unsqueeze_286 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_70: "f32[32, 128, 5, 5]" = torch.ops.aten.relu.default(add_357);  add_357 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_71: "f32[32, 768, 1, 1]" = torch.ops.aten.convolution.default(relu_70, primals_214, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_358: "i64[]" = torch.ops.aten.add.Tensor(primals_508, 1)
        var_mean_71 = torch.ops.aten.var_mean.correction(convolution_71, [0, 2, 3], correction = 0, keepdim = True)
        getitem_148: "f32[1, 768, 1, 1]" = var_mean_71[0]
        getitem_149: "f32[1, 768, 1, 1]" = var_mean_71[1];  var_mean_71 = None
        add_359: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_148, 0.001)
        rsqrt_71: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_359);  add_359 = None
        sub_71: "f32[32, 768, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_71, getitem_149)
        mul_500: "f32[32, 768, 1, 1]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = None
        squeeze_213: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_149, [0, 2, 3]);  getitem_149 = None
        squeeze_214: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_71, [0, 2, 3]);  rsqrt_71 = None
        mul_501: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_213, 0.1)
        mul_502: "f32[768]" = torch.ops.aten.mul.Tensor(primals_506, 0.9)
        add_360: "f32[768]" = torch.ops.aten.add.Tensor(mul_501, mul_502);  mul_501 = mul_502 = None
        squeeze_215: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_148, [0, 2, 3]);  getitem_148 = None
        mul_503: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_215, 1.032258064516129);  squeeze_215 = None
        mul_504: "f32[768]" = torch.ops.aten.mul.Tensor(mul_503, 0.1);  mul_503 = None
        mul_505: "f32[768]" = torch.ops.aten.mul.Tensor(primals_507, 0.9)
        add_361: "f32[768]" = torch.ops.aten.add.Tensor(mul_504, mul_505);  mul_504 = mul_505 = None
        unsqueeze_287: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_215, -1)
        unsqueeze_288: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, -1);  unsqueeze_287 = None
        mul_506: "f32[32, 768, 1, 1]" = torch.ops.aten.mul.Tensor(mul_500, unsqueeze_288);  mul_500 = unsqueeze_288 = None
        unsqueeze_289: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_216, -1);  primals_216 = None
        unsqueeze_290: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, -1);  unsqueeze_289 = None
        add_362: "f32[32, 768, 1, 1]" = torch.ops.aten.add.Tensor(mul_506, unsqueeze_290);  mul_506 = unsqueeze_290 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_71: "f32[32, 768, 1, 1]" = torch.ops.aten.relu.default(add_362);  add_362 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:389 in forward, code: x = F.adaptive_avg_pool2d(x, (1, 1))
        mean: "f32[32, 768, 1, 1]" = torch.ops.aten.mean.dim(relu_71, [-1, -2], True)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:391 in forward, code: x = torch.flatten(x, 1)
        view: "f32[32, 768]" = torch.ops.aten.view.default(mean, [32, 768]);  mean = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:393 in forward, code: x = self.fc(x)
        permute: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_217, [1, 0]);  primals_217 = None
        addmm: "f32[32, 1000]" = torch.ops.aten.addmm.default(primals_218, view, permute);  primals_218 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_72: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(cat_8, primals_219, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_363: "i64[]" = torch.ops.aten.add.Tensor(primals_511, 1)
        var_mean_72 = torch.ops.aten.var_mean.correction(convolution_72, [0, 2, 3], correction = 0, keepdim = True)
        getitem_150: "f32[1, 192, 1, 1]" = var_mean_72[0]
        getitem_151: "f32[1, 192, 1, 1]" = var_mean_72[1];  var_mean_72 = None
        add_364: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_150, 0.001)
        rsqrt_72: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_364);  add_364 = None
        sub_72: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_72, getitem_151)
        mul_507: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = None
        squeeze_216: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_151, [0, 2, 3]);  getitem_151 = None
        squeeze_217: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_72, [0, 2, 3]);  rsqrt_72 = None
        mul_508: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_216, 0.1)
        mul_509: "f32[192]" = torch.ops.aten.mul.Tensor(primals_509, 0.9)
        add_365: "f32[192]" = torch.ops.aten.add.Tensor(mul_508, mul_509);  mul_508 = mul_509 = None
        squeeze_218: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_150, [0, 2, 3]);  getitem_150 = None
        mul_510: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_218, 1.0001081431815724);  squeeze_218 = None
        mul_511: "f32[192]" = torch.ops.aten.mul.Tensor(mul_510, 0.1);  mul_510 = None
        mul_512: "f32[192]" = torch.ops.aten.mul.Tensor(primals_510, 0.9)
        add_366: "f32[192]" = torch.ops.aten.add.Tensor(mul_511, mul_512);  mul_511 = mul_512 = None
        unsqueeze_291: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_220, -1)
        unsqueeze_292: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, -1);  unsqueeze_291 = None
        mul_513: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_507, unsqueeze_292);  mul_507 = unsqueeze_292 = None
        unsqueeze_293: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_221, -1);  primals_221 = None
        unsqueeze_294: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, -1);  unsqueeze_293 = None
        add_367: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_513, unsqueeze_294);  mul_513 = unsqueeze_294 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_72: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_367);  add_367 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_73: "f32[32, 320, 8, 8]" = torch.ops.aten.convolution.default(relu_72, primals_222, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_368: "i64[]" = torch.ops.aten.add.Tensor(primals_514, 1)
        var_mean_73 = torch.ops.aten.var_mean.correction(convolution_73, [0, 2, 3], correction = 0, keepdim = True)
        getitem_152: "f32[1, 320, 1, 1]" = var_mean_73[0]
        getitem_153: "f32[1, 320, 1, 1]" = var_mean_73[1];  var_mean_73 = None
        add_369: "f32[1, 320, 1, 1]" = torch.ops.aten.add.Tensor(getitem_152, 0.001)
        rsqrt_73: "f32[1, 320, 1, 1]" = torch.ops.aten.rsqrt.default(add_369);  add_369 = None
        sub_73: "f32[32, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_73, getitem_153)
        mul_514: "f32[32, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = None
        squeeze_219: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_153, [0, 2, 3]);  getitem_153 = None
        squeeze_220: "f32[320]" = torch.ops.aten.squeeze.dims(rsqrt_73, [0, 2, 3]);  rsqrt_73 = None
        mul_515: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_219, 0.1)
        mul_516: "f32[320]" = torch.ops.aten.mul.Tensor(primals_512, 0.9)
        add_370: "f32[320]" = torch.ops.aten.add.Tensor(mul_515, mul_516);  mul_515 = mul_516 = None
        squeeze_221: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_152, [0, 2, 3]);  getitem_152 = None
        mul_517: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_221, 1.0004885197850513);  squeeze_221 = None
        mul_518: "f32[320]" = torch.ops.aten.mul.Tensor(mul_517, 0.1);  mul_517 = None
        mul_519: "f32[320]" = torch.ops.aten.mul.Tensor(primals_513, 0.9)
        add_371: "f32[320]" = torch.ops.aten.add.Tensor(mul_518, mul_519);  mul_518 = mul_519 = None
        unsqueeze_295: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_223, -1)
        unsqueeze_296: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, -1);  unsqueeze_295 = None
        mul_520: "f32[32, 320, 8, 8]" = torch.ops.aten.mul.Tensor(mul_514, unsqueeze_296);  mul_514 = unsqueeze_296 = None
        unsqueeze_297: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_224, -1);  primals_224 = None
        unsqueeze_298: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, -1);  unsqueeze_297 = None
        add_372: "f32[32, 320, 8, 8]" = torch.ops.aten.add.Tensor(mul_520, unsqueeze_298);  mul_520 = unsqueeze_298 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_73: "f32[32, 320, 8, 8]" = torch.ops.aten.relu.default(add_372);  add_372 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_74: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(cat_8, primals_225, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_373: "i64[]" = torch.ops.aten.add.Tensor(primals_517, 1)
        var_mean_74 = torch.ops.aten.var_mean.correction(convolution_74, [0, 2, 3], correction = 0, keepdim = True)
        getitem_154: "f32[1, 192, 1, 1]" = var_mean_74[0]
        getitem_155: "f32[1, 192, 1, 1]" = var_mean_74[1];  var_mean_74 = None
        add_374: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_154, 0.001)
        rsqrt_74: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_374);  add_374 = None
        sub_74: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_74, getitem_155)
        mul_521: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = None
        squeeze_222: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_155, [0, 2, 3]);  getitem_155 = None
        squeeze_223: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_74, [0, 2, 3]);  rsqrt_74 = None
        mul_522: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_222, 0.1)
        mul_523: "f32[192]" = torch.ops.aten.mul.Tensor(primals_515, 0.9)
        add_375: "f32[192]" = torch.ops.aten.add.Tensor(mul_522, mul_523);  mul_522 = mul_523 = None
        squeeze_224: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_154, [0, 2, 3]);  getitem_154 = None
        mul_524: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_224, 1.0001081431815724);  squeeze_224 = None
        mul_525: "f32[192]" = torch.ops.aten.mul.Tensor(mul_524, 0.1);  mul_524 = None
        mul_526: "f32[192]" = torch.ops.aten.mul.Tensor(primals_516, 0.9)
        add_376: "f32[192]" = torch.ops.aten.add.Tensor(mul_525, mul_526);  mul_525 = mul_526 = None
        unsqueeze_299: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_226, -1)
        unsqueeze_300: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, -1);  unsqueeze_299 = None
        mul_527: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_521, unsqueeze_300);  mul_521 = unsqueeze_300 = None
        unsqueeze_301: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_227, -1);  primals_227 = None
        unsqueeze_302: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, -1);  unsqueeze_301 = None
        add_377: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_527, unsqueeze_302);  mul_527 = unsqueeze_302 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_74: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_377);  add_377 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_75: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_74, primals_228, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_378: "i64[]" = torch.ops.aten.add.Tensor(primals_520, 1)
        var_mean_75 = torch.ops.aten.var_mean.correction(convolution_75, [0, 2, 3], correction = 0, keepdim = True)
        getitem_156: "f32[1, 192, 1, 1]" = var_mean_75[0]
        getitem_157: "f32[1, 192, 1, 1]" = var_mean_75[1];  var_mean_75 = None
        add_379: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_156, 0.001)
        rsqrt_75: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_379);  add_379 = None
        sub_75: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_75, getitem_157)
        mul_528: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = None
        squeeze_225: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
        squeeze_226: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_75, [0, 2, 3]);  rsqrt_75 = None
        mul_529: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_225, 0.1)
        mul_530: "f32[192]" = torch.ops.aten.mul.Tensor(primals_518, 0.9)
        add_380: "f32[192]" = torch.ops.aten.add.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
        squeeze_227: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_156, [0, 2, 3]);  getitem_156 = None
        mul_531: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_227, 1.0001081431815724);  squeeze_227 = None
        mul_532: "f32[192]" = torch.ops.aten.mul.Tensor(mul_531, 0.1);  mul_531 = None
        mul_533: "f32[192]" = torch.ops.aten.mul.Tensor(primals_519, 0.9)
        add_381: "f32[192]" = torch.ops.aten.add.Tensor(mul_532, mul_533);  mul_532 = mul_533 = None
        unsqueeze_303: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_229, -1)
        unsqueeze_304: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, -1);  unsqueeze_303 = None
        mul_534: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_528, unsqueeze_304);  mul_528 = unsqueeze_304 = None
        unsqueeze_305: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_230, -1);  primals_230 = None
        unsqueeze_306: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, -1);  unsqueeze_305 = None
        add_382: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_534, unsqueeze_306);  mul_534 = unsqueeze_306 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_75: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_382);  add_382 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_76: "f32[32, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_75, primals_231, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_383: "i64[]" = torch.ops.aten.add.Tensor(primals_523, 1)
        var_mean_76 = torch.ops.aten.var_mean.correction(convolution_76, [0, 2, 3], correction = 0, keepdim = True)
        getitem_158: "f32[1, 192, 1, 1]" = var_mean_76[0]
        getitem_159: "f32[1, 192, 1, 1]" = var_mean_76[1];  var_mean_76 = None
        add_384: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_158, 0.001)
        rsqrt_76: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_384);  add_384 = None
        sub_76: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_76, getitem_159)
        mul_535: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = None
        squeeze_228: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_159, [0, 2, 3]);  getitem_159 = None
        squeeze_229: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_76, [0, 2, 3]);  rsqrt_76 = None
        mul_536: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_228, 0.1)
        mul_537: "f32[192]" = torch.ops.aten.mul.Tensor(primals_521, 0.9)
        add_385: "f32[192]" = torch.ops.aten.add.Tensor(mul_536, mul_537);  mul_536 = mul_537 = None
        squeeze_230: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_158, [0, 2, 3]);  getitem_158 = None
        mul_538: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_230, 1.0001081431815724);  squeeze_230 = None
        mul_539: "f32[192]" = torch.ops.aten.mul.Tensor(mul_538, 0.1);  mul_538 = None
        mul_540: "f32[192]" = torch.ops.aten.mul.Tensor(primals_522, 0.9)
        add_386: "f32[192]" = torch.ops.aten.add.Tensor(mul_539, mul_540);  mul_539 = mul_540 = None
        unsqueeze_307: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_232, -1)
        unsqueeze_308: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, -1);  unsqueeze_307 = None
        mul_541: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_535, unsqueeze_308);  mul_535 = unsqueeze_308 = None
        unsqueeze_309: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_233, -1);  primals_233 = None
        unsqueeze_310: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, -1);  unsqueeze_309 = None
        add_387: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_541, unsqueeze_310);  mul_541 = unsqueeze_310 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_76: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_387);  add_387 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_77: "f32[32, 192, 8, 8]" = torch.ops.aten.convolution.default(relu_76, primals_234, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_388: "i64[]" = torch.ops.aten.add.Tensor(primals_526, 1)
        var_mean_77 = torch.ops.aten.var_mean.correction(convolution_77, [0, 2, 3], correction = 0, keepdim = True)
        getitem_160: "f32[1, 192, 1, 1]" = var_mean_77[0]
        getitem_161: "f32[1, 192, 1, 1]" = var_mean_77[1];  var_mean_77 = None
        add_389: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_160, 0.001)
        rsqrt_77: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_389);  add_389 = None
        sub_77: "f32[32, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_77, getitem_161)
        mul_542: "f32[32, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = None
        squeeze_231: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_161, [0, 2, 3]);  getitem_161 = None
        squeeze_232: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_77, [0, 2, 3]);  rsqrt_77 = None
        mul_543: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_231, 0.1)
        mul_544: "f32[192]" = torch.ops.aten.mul.Tensor(primals_524, 0.9)
        add_390: "f32[192]" = torch.ops.aten.add.Tensor(mul_543, mul_544);  mul_543 = mul_544 = None
        squeeze_233: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_160, [0, 2, 3]);  getitem_160 = None
        mul_545: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_233, 1.0004885197850513);  squeeze_233 = None
        mul_546: "f32[192]" = torch.ops.aten.mul.Tensor(mul_545, 0.1);  mul_545 = None
        mul_547: "f32[192]" = torch.ops.aten.mul.Tensor(primals_525, 0.9)
        add_391: "f32[192]" = torch.ops.aten.add.Tensor(mul_546, mul_547);  mul_546 = mul_547 = None
        unsqueeze_311: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_235, -1)
        unsqueeze_312: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, -1);  unsqueeze_311 = None
        mul_548: "f32[32, 192, 8, 8]" = torch.ops.aten.mul.Tensor(mul_542, unsqueeze_312);  mul_542 = unsqueeze_312 = None
        unsqueeze_313: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_236, -1);  primals_236 = None
        unsqueeze_314: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, -1);  unsqueeze_313 = None
        add_392: "f32[32, 192, 8, 8]" = torch.ops.aten.add.Tensor(mul_548, unsqueeze_314);  mul_548 = unsqueeze_314 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_77: "f32[32, 192, 8, 8]" = torch.ops.aten.relu.default(add_392);  add_392 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:311 in _forward, code: branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        max_pool2d_with_indices_3 = torch.ops.aten.max_pool2d_with_indices.default(cat_8, [3, 3], [2, 2])
        getitem_162: "f32[32, 768, 8, 8]" = max_pool2d_with_indices_3[0]
        getitem_163: "i64[32, 768, 8, 8]" = max_pool2d_with_indices_3[1];  max_pool2d_with_indices_3 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:317 in forward, code: return torch.cat(outputs, 1)
        cat_9: "f32[32, 1280, 8, 8]" = torch.ops.aten.cat.default([relu_73, relu_77, getitem_162], 1);  getitem_162 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_78: "f32[32, 320, 8, 8]" = torch.ops.aten.convolution.default(cat_9, primals_237, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_393: "i64[]" = torch.ops.aten.add.Tensor(primals_529, 1)
        var_mean_78 = torch.ops.aten.var_mean.correction(convolution_78, [0, 2, 3], correction = 0, keepdim = True)
        getitem_164: "f32[1, 320, 1, 1]" = var_mean_78[0]
        getitem_165: "f32[1, 320, 1, 1]" = var_mean_78[1];  var_mean_78 = None
        add_394: "f32[1, 320, 1, 1]" = torch.ops.aten.add.Tensor(getitem_164, 0.001)
        rsqrt_78: "f32[1, 320, 1, 1]" = torch.ops.aten.rsqrt.default(add_394);  add_394 = None
        sub_78: "f32[32, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_78, getitem_165)
        mul_549: "f32[32, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = None
        squeeze_234: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_165, [0, 2, 3]);  getitem_165 = None
        squeeze_235: "f32[320]" = torch.ops.aten.squeeze.dims(rsqrt_78, [0, 2, 3]);  rsqrt_78 = None
        mul_550: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_234, 0.1)
        mul_551: "f32[320]" = torch.ops.aten.mul.Tensor(primals_527, 0.9)
        add_395: "f32[320]" = torch.ops.aten.add.Tensor(mul_550, mul_551);  mul_550 = mul_551 = None
        squeeze_236: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_164, [0, 2, 3]);  getitem_164 = None
        mul_552: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_236, 1.0004885197850513);  squeeze_236 = None
        mul_553: "f32[320]" = torch.ops.aten.mul.Tensor(mul_552, 0.1);  mul_552 = None
        mul_554: "f32[320]" = torch.ops.aten.mul.Tensor(primals_528, 0.9)
        add_396: "f32[320]" = torch.ops.aten.add.Tensor(mul_553, mul_554);  mul_553 = mul_554 = None
        unsqueeze_315: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_238, -1)
        unsqueeze_316: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, -1);  unsqueeze_315 = None
        mul_555: "f32[32, 320, 8, 8]" = torch.ops.aten.mul.Tensor(mul_549, unsqueeze_316);  mul_549 = unsqueeze_316 = None
        unsqueeze_317: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_239, -1);  primals_239 = None
        unsqueeze_318: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, -1);  unsqueeze_317 = None
        add_397: "f32[32, 320, 8, 8]" = torch.ops.aten.add.Tensor(mul_555, unsqueeze_318);  mul_555 = unsqueeze_318 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_78: "f32[32, 320, 8, 8]" = torch.ops.aten.relu.default(add_397);  add_397 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_79: "f32[32, 384, 8, 8]" = torch.ops.aten.convolution.default(cat_9, primals_240, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_398: "i64[]" = torch.ops.aten.add.Tensor(primals_532, 1)
        var_mean_79 = torch.ops.aten.var_mean.correction(convolution_79, [0, 2, 3], correction = 0, keepdim = True)
        getitem_166: "f32[1, 384, 1, 1]" = var_mean_79[0]
        getitem_167: "f32[1, 384, 1, 1]" = var_mean_79[1];  var_mean_79 = None
        add_399: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_166, 0.001)
        rsqrt_79: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_399);  add_399 = None
        sub_79: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_79, getitem_167)
        mul_556: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = None
        squeeze_237: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_167, [0, 2, 3]);  getitem_167 = None
        squeeze_238: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_79, [0, 2, 3]);  rsqrt_79 = None
        mul_557: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_237, 0.1)
        mul_558: "f32[384]" = torch.ops.aten.mul.Tensor(primals_530, 0.9)
        add_400: "f32[384]" = torch.ops.aten.add.Tensor(mul_557, mul_558);  mul_557 = mul_558 = None
        squeeze_239: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_166, [0, 2, 3]);  getitem_166 = None
        mul_559: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_239, 1.0004885197850513);  squeeze_239 = None
        mul_560: "f32[384]" = torch.ops.aten.mul.Tensor(mul_559, 0.1);  mul_559 = None
        mul_561: "f32[384]" = torch.ops.aten.mul.Tensor(primals_531, 0.9)
        add_401: "f32[384]" = torch.ops.aten.add.Tensor(mul_560, mul_561);  mul_560 = mul_561 = None
        unsqueeze_319: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_241, -1)
        unsqueeze_320: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, -1);  unsqueeze_319 = None
        mul_562: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_556, unsqueeze_320);  mul_556 = unsqueeze_320 = None
        unsqueeze_321: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_242, -1);  primals_242 = None
        unsqueeze_322: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, -1);  unsqueeze_321 = None
        add_402: "f32[32, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_562, unsqueeze_322);  mul_562 = unsqueeze_322 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_79: "f32[32, 384, 8, 8]" = torch.ops.aten.relu.default(add_402);  add_402 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_80: "f32[32, 384, 8, 8]" = torch.ops.aten.convolution.default(relu_79, primals_243, None, [1, 1], [0, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_403: "i64[]" = torch.ops.aten.add.Tensor(primals_535, 1)
        var_mean_80 = torch.ops.aten.var_mean.correction(convolution_80, [0, 2, 3], correction = 0, keepdim = True)
        getitem_168: "f32[1, 384, 1, 1]" = var_mean_80[0]
        getitem_169: "f32[1, 384, 1, 1]" = var_mean_80[1];  var_mean_80 = None
        add_404: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_168, 0.001)
        rsqrt_80: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_404);  add_404 = None
        sub_80: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_80, getitem_169)
        mul_563: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = None
        squeeze_240: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_169, [0, 2, 3]);  getitem_169 = None
        squeeze_241: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_80, [0, 2, 3]);  rsqrt_80 = None
        mul_564: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_240, 0.1)
        mul_565: "f32[384]" = torch.ops.aten.mul.Tensor(primals_533, 0.9)
        add_405: "f32[384]" = torch.ops.aten.add.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
        squeeze_242: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_168, [0, 2, 3]);  getitem_168 = None
        mul_566: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_242, 1.0004885197850513);  squeeze_242 = None
        mul_567: "f32[384]" = torch.ops.aten.mul.Tensor(mul_566, 0.1);  mul_566 = None
        mul_568: "f32[384]" = torch.ops.aten.mul.Tensor(primals_534, 0.9)
        add_406: "f32[384]" = torch.ops.aten.add.Tensor(mul_567, mul_568);  mul_567 = mul_568 = None
        unsqueeze_323: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_244, -1)
        unsqueeze_324: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, -1);  unsqueeze_323 = None
        mul_569: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_563, unsqueeze_324);  mul_563 = unsqueeze_324 = None
        unsqueeze_325: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_245, -1);  primals_245 = None
        unsqueeze_326: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, -1);  unsqueeze_325 = None
        add_407: "f32[32, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_569, unsqueeze_326);  mul_569 = unsqueeze_326 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_80: "f32[32, 384, 8, 8]" = torch.ops.aten.relu.default(add_407);  add_407 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_81: "f32[32, 384, 8, 8]" = torch.ops.aten.convolution.default(relu_79, primals_246, None, [1, 1], [1, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_408: "i64[]" = torch.ops.aten.add.Tensor(primals_538, 1)
        var_mean_81 = torch.ops.aten.var_mean.correction(convolution_81, [0, 2, 3], correction = 0, keepdim = True)
        getitem_170: "f32[1, 384, 1, 1]" = var_mean_81[0]
        getitem_171: "f32[1, 384, 1, 1]" = var_mean_81[1];  var_mean_81 = None
        add_409: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_170, 0.001)
        rsqrt_81: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_409);  add_409 = None
        sub_81: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_81, getitem_171)
        mul_570: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = None
        squeeze_243: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_171, [0, 2, 3]);  getitem_171 = None
        squeeze_244: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_81, [0, 2, 3]);  rsqrt_81 = None
        mul_571: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_243, 0.1)
        mul_572: "f32[384]" = torch.ops.aten.mul.Tensor(primals_536, 0.9)
        add_410: "f32[384]" = torch.ops.aten.add.Tensor(mul_571, mul_572);  mul_571 = mul_572 = None
        squeeze_245: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_170, [0, 2, 3]);  getitem_170 = None
        mul_573: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_245, 1.0004885197850513);  squeeze_245 = None
        mul_574: "f32[384]" = torch.ops.aten.mul.Tensor(mul_573, 0.1);  mul_573 = None
        mul_575: "f32[384]" = torch.ops.aten.mul.Tensor(primals_537, 0.9)
        add_411: "f32[384]" = torch.ops.aten.add.Tensor(mul_574, mul_575);  mul_574 = mul_575 = None
        unsqueeze_327: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_247, -1)
        unsqueeze_328: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, -1);  unsqueeze_327 = None
        mul_576: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_570, unsqueeze_328);  mul_570 = unsqueeze_328 = None
        unsqueeze_329: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_248, -1);  primals_248 = None
        unsqueeze_330: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, -1);  unsqueeze_329 = None
        add_412: "f32[32, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_576, unsqueeze_330);  mul_576 = unsqueeze_330 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_81: "f32[32, 384, 8, 8]" = torch.ops.aten.relu.default(add_412);  add_412 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:346 in _forward, code: branch3x3 = torch.cat(branch3x3, 1)
        cat_10: "f32[32, 768, 8, 8]" = torch.ops.aten.cat.default([relu_80, relu_81], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_82: "f32[32, 448, 8, 8]" = torch.ops.aten.convolution.default(cat_9, primals_249, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_413: "i64[]" = torch.ops.aten.add.Tensor(primals_541, 1)
        var_mean_82 = torch.ops.aten.var_mean.correction(convolution_82, [0, 2, 3], correction = 0, keepdim = True)
        getitem_172: "f32[1, 448, 1, 1]" = var_mean_82[0]
        getitem_173: "f32[1, 448, 1, 1]" = var_mean_82[1];  var_mean_82 = None
        add_414: "f32[1, 448, 1, 1]" = torch.ops.aten.add.Tensor(getitem_172, 0.001)
        rsqrt_82: "f32[1, 448, 1, 1]" = torch.ops.aten.rsqrt.default(add_414);  add_414 = None
        sub_82: "f32[32, 448, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_82, getitem_173)
        mul_577: "f32[32, 448, 8, 8]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = None
        squeeze_246: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_173, [0, 2, 3]);  getitem_173 = None
        squeeze_247: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_82, [0, 2, 3]);  rsqrt_82 = None
        mul_578: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_246, 0.1)
        mul_579: "f32[448]" = torch.ops.aten.mul.Tensor(primals_539, 0.9)
        add_415: "f32[448]" = torch.ops.aten.add.Tensor(mul_578, mul_579);  mul_578 = mul_579 = None
        squeeze_248: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_172, [0, 2, 3]);  getitem_172 = None
        mul_580: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_248, 1.0004885197850513);  squeeze_248 = None
        mul_581: "f32[448]" = torch.ops.aten.mul.Tensor(mul_580, 0.1);  mul_580 = None
        mul_582: "f32[448]" = torch.ops.aten.mul.Tensor(primals_540, 0.9)
        add_416: "f32[448]" = torch.ops.aten.add.Tensor(mul_581, mul_582);  mul_581 = mul_582 = None
        unsqueeze_331: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_250, -1)
        unsqueeze_332: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, -1);  unsqueeze_331 = None
        mul_583: "f32[32, 448, 8, 8]" = torch.ops.aten.mul.Tensor(mul_577, unsqueeze_332);  mul_577 = unsqueeze_332 = None
        unsqueeze_333: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_251, -1);  primals_251 = None
        unsqueeze_334: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, -1);  unsqueeze_333 = None
        add_417: "f32[32, 448, 8, 8]" = torch.ops.aten.add.Tensor(mul_583, unsqueeze_334);  mul_583 = unsqueeze_334 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_82: "f32[32, 448, 8, 8]" = torch.ops.aten.relu.default(add_417);  add_417 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_83: "f32[32, 384, 8, 8]" = torch.ops.aten.convolution.default(relu_82, primals_252, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_418: "i64[]" = torch.ops.aten.add.Tensor(primals_544, 1)
        var_mean_83 = torch.ops.aten.var_mean.correction(convolution_83, [0, 2, 3], correction = 0, keepdim = True)
        getitem_174: "f32[1, 384, 1, 1]" = var_mean_83[0]
        getitem_175: "f32[1, 384, 1, 1]" = var_mean_83[1];  var_mean_83 = None
        add_419: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_174, 0.001)
        rsqrt_83: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_419);  add_419 = None
        sub_83: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_83, getitem_175)
        mul_584: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = None
        squeeze_249: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_175, [0, 2, 3]);  getitem_175 = None
        squeeze_250: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_83, [0, 2, 3]);  rsqrt_83 = None
        mul_585: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_249, 0.1)
        mul_586: "f32[384]" = torch.ops.aten.mul.Tensor(primals_542, 0.9)
        add_420: "f32[384]" = torch.ops.aten.add.Tensor(mul_585, mul_586);  mul_585 = mul_586 = None
        squeeze_251: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_174, [0, 2, 3]);  getitem_174 = None
        mul_587: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_251, 1.0004885197850513);  squeeze_251 = None
        mul_588: "f32[384]" = torch.ops.aten.mul.Tensor(mul_587, 0.1);  mul_587 = None
        mul_589: "f32[384]" = torch.ops.aten.mul.Tensor(primals_543, 0.9)
        add_421: "f32[384]" = torch.ops.aten.add.Tensor(mul_588, mul_589);  mul_588 = mul_589 = None
        unsqueeze_335: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_253, -1)
        unsqueeze_336: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, -1);  unsqueeze_335 = None
        mul_590: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_584, unsqueeze_336);  mul_584 = unsqueeze_336 = None
        unsqueeze_337: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_254, -1);  primals_254 = None
        unsqueeze_338: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, -1);  unsqueeze_337 = None
        add_422: "f32[32, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_590, unsqueeze_338);  mul_590 = unsqueeze_338 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_83: "f32[32, 384, 8, 8]" = torch.ops.aten.relu.default(add_422);  add_422 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_84: "f32[32, 384, 8, 8]" = torch.ops.aten.convolution.default(relu_83, primals_255, None, [1, 1], [0, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_423: "i64[]" = torch.ops.aten.add.Tensor(primals_547, 1)
        var_mean_84 = torch.ops.aten.var_mean.correction(convolution_84, [0, 2, 3], correction = 0, keepdim = True)
        getitem_176: "f32[1, 384, 1, 1]" = var_mean_84[0]
        getitem_177: "f32[1, 384, 1, 1]" = var_mean_84[1];  var_mean_84 = None
        add_424: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_176, 0.001)
        rsqrt_84: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_424);  add_424 = None
        sub_84: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_84, getitem_177)
        mul_591: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = None
        squeeze_252: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_177, [0, 2, 3]);  getitem_177 = None
        squeeze_253: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_84, [0, 2, 3]);  rsqrt_84 = None
        mul_592: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_252, 0.1)
        mul_593: "f32[384]" = torch.ops.aten.mul.Tensor(primals_545, 0.9)
        add_425: "f32[384]" = torch.ops.aten.add.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
        squeeze_254: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_176, [0, 2, 3]);  getitem_176 = None
        mul_594: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_254, 1.0004885197850513);  squeeze_254 = None
        mul_595: "f32[384]" = torch.ops.aten.mul.Tensor(mul_594, 0.1);  mul_594 = None
        mul_596: "f32[384]" = torch.ops.aten.mul.Tensor(primals_546, 0.9)
        add_426: "f32[384]" = torch.ops.aten.add.Tensor(mul_595, mul_596);  mul_595 = mul_596 = None
        unsqueeze_339: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_256, -1)
        unsqueeze_340: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, -1);  unsqueeze_339 = None
        mul_597: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_591, unsqueeze_340);  mul_591 = unsqueeze_340 = None
        unsqueeze_341: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_257, -1);  primals_257 = None
        unsqueeze_342: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, -1);  unsqueeze_341 = None
        add_427: "f32[32, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_597, unsqueeze_342);  mul_597 = unsqueeze_342 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_84: "f32[32, 384, 8, 8]" = torch.ops.aten.relu.default(add_427);  add_427 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_85: "f32[32, 384, 8, 8]" = torch.ops.aten.convolution.default(relu_83, primals_258, None, [1, 1], [1, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_428: "i64[]" = torch.ops.aten.add.Tensor(primals_550, 1)
        var_mean_85 = torch.ops.aten.var_mean.correction(convolution_85, [0, 2, 3], correction = 0, keepdim = True)
        getitem_178: "f32[1, 384, 1, 1]" = var_mean_85[0]
        getitem_179: "f32[1, 384, 1, 1]" = var_mean_85[1];  var_mean_85 = None
        add_429: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_178, 0.001)
        rsqrt_85: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_429);  add_429 = None
        sub_85: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_85, getitem_179)
        mul_598: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_85);  sub_85 = None
        squeeze_255: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_179, [0, 2, 3]);  getitem_179 = None
        squeeze_256: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_85, [0, 2, 3]);  rsqrt_85 = None
        mul_599: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_255, 0.1)
        mul_600: "f32[384]" = torch.ops.aten.mul.Tensor(primals_548, 0.9)
        add_430: "f32[384]" = torch.ops.aten.add.Tensor(mul_599, mul_600);  mul_599 = mul_600 = None
        squeeze_257: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_178, [0, 2, 3]);  getitem_178 = None
        mul_601: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_257, 1.0004885197850513);  squeeze_257 = None
        mul_602: "f32[384]" = torch.ops.aten.mul.Tensor(mul_601, 0.1);  mul_601 = None
        mul_603: "f32[384]" = torch.ops.aten.mul.Tensor(primals_549, 0.9)
        add_431: "f32[384]" = torch.ops.aten.add.Tensor(mul_602, mul_603);  mul_602 = mul_603 = None
        unsqueeze_343: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_259, -1)
        unsqueeze_344: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, -1);  unsqueeze_343 = None
        mul_604: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_598, unsqueeze_344);  mul_598 = unsqueeze_344 = None
        unsqueeze_345: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_260, -1);  primals_260 = None
        unsqueeze_346: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, -1);  unsqueeze_345 = None
        add_432: "f32[32, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_604, unsqueeze_346);  mul_604 = unsqueeze_346 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_85: "f32[32, 384, 8, 8]" = torch.ops.aten.relu.default(add_432);  add_432 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:354 in _forward, code: branch3x3dbl = torch.cat(branch3x3dbl, 1)
        cat_11: "f32[32, 768, 8, 8]" = torch.ops.aten.cat.default([relu_84, relu_85], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:356 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_8: "f32[32, 1280, 8, 8]" = torch.ops.aten.avg_pool2d.default(cat_9, [3, 3], [1, 1], [1, 1])
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_86: "f32[32, 192, 8, 8]" = torch.ops.aten.convolution.default(avg_pool2d_8, primals_261, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_433: "i64[]" = torch.ops.aten.add.Tensor(primals_553, 1)
        var_mean_86 = torch.ops.aten.var_mean.correction(convolution_86, [0, 2, 3], correction = 0, keepdim = True)
        getitem_180: "f32[1, 192, 1, 1]" = var_mean_86[0]
        getitem_181: "f32[1, 192, 1, 1]" = var_mean_86[1];  var_mean_86 = None
        add_434: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_180, 0.001)
        rsqrt_86: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_434);  add_434 = None
        sub_86: "f32[32, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_86, getitem_181)
        mul_605: "f32[32, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_86);  sub_86 = None
        squeeze_258: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_181, [0, 2, 3]);  getitem_181 = None
        squeeze_259: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_86, [0, 2, 3]);  rsqrt_86 = None
        mul_606: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_258, 0.1)
        mul_607: "f32[192]" = torch.ops.aten.mul.Tensor(primals_551, 0.9)
        add_435: "f32[192]" = torch.ops.aten.add.Tensor(mul_606, mul_607);  mul_606 = mul_607 = None
        squeeze_260: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_180, [0, 2, 3]);  getitem_180 = None
        mul_608: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_260, 1.0004885197850513);  squeeze_260 = None
        mul_609: "f32[192]" = torch.ops.aten.mul.Tensor(mul_608, 0.1);  mul_608 = None
        mul_610: "f32[192]" = torch.ops.aten.mul.Tensor(primals_552, 0.9)
        add_436: "f32[192]" = torch.ops.aten.add.Tensor(mul_609, mul_610);  mul_609 = mul_610 = None
        unsqueeze_347: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_262, -1)
        unsqueeze_348: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, -1);  unsqueeze_347 = None
        mul_611: "f32[32, 192, 8, 8]" = torch.ops.aten.mul.Tensor(mul_605, unsqueeze_348);  mul_605 = unsqueeze_348 = None
        unsqueeze_349: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_263, -1);  primals_263 = None
        unsqueeze_350: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, -1);  unsqueeze_349 = None
        add_437: "f32[32, 192, 8, 8]" = torch.ops.aten.add.Tensor(mul_611, unsqueeze_350);  mul_611 = unsqueeze_350 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_86: "f32[32, 192, 8, 8]" = torch.ops.aten.relu.default(add_437);  add_437 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:364 in forward, code: return torch.cat(outputs, 1)
        cat_12: "f32[32, 2048, 8, 8]" = torch.ops.aten.cat.default([relu_78, cat_10, cat_11, relu_86], 1);  cat_10 = cat_11 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_87: "f32[32, 320, 8, 8]" = torch.ops.aten.convolution.default(cat_12, primals_264, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_438: "i64[]" = torch.ops.aten.add.Tensor(primals_556, 1)
        var_mean_87 = torch.ops.aten.var_mean.correction(convolution_87, [0, 2, 3], correction = 0, keepdim = True)
        getitem_182: "f32[1, 320, 1, 1]" = var_mean_87[0]
        getitem_183: "f32[1, 320, 1, 1]" = var_mean_87[1];  var_mean_87 = None
        add_439: "f32[1, 320, 1, 1]" = torch.ops.aten.add.Tensor(getitem_182, 0.001)
        rsqrt_87: "f32[1, 320, 1, 1]" = torch.ops.aten.rsqrt.default(add_439);  add_439 = None
        sub_87: "f32[32, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_87, getitem_183)
        mul_612: "f32[32, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_87);  sub_87 = None
        squeeze_261: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_183, [0, 2, 3]);  getitem_183 = None
        squeeze_262: "f32[320]" = torch.ops.aten.squeeze.dims(rsqrt_87, [0, 2, 3]);  rsqrt_87 = None
        mul_613: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_261, 0.1)
        mul_614: "f32[320]" = torch.ops.aten.mul.Tensor(primals_554, 0.9)
        add_440: "f32[320]" = torch.ops.aten.add.Tensor(mul_613, mul_614);  mul_613 = mul_614 = None
        squeeze_263: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_182, [0, 2, 3]);  getitem_182 = None
        mul_615: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_263, 1.0004885197850513);  squeeze_263 = None
        mul_616: "f32[320]" = torch.ops.aten.mul.Tensor(mul_615, 0.1);  mul_615 = None
        mul_617: "f32[320]" = torch.ops.aten.mul.Tensor(primals_555, 0.9)
        add_441: "f32[320]" = torch.ops.aten.add.Tensor(mul_616, mul_617);  mul_616 = mul_617 = None
        unsqueeze_351: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_265, -1)
        unsqueeze_352: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, -1);  unsqueeze_351 = None
        mul_618: "f32[32, 320, 8, 8]" = torch.ops.aten.mul.Tensor(mul_612, unsqueeze_352);  mul_612 = unsqueeze_352 = None
        unsqueeze_353: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_266, -1);  primals_266 = None
        unsqueeze_354: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, -1);  unsqueeze_353 = None
        add_442: "f32[32, 320, 8, 8]" = torch.ops.aten.add.Tensor(mul_618, unsqueeze_354);  mul_618 = unsqueeze_354 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_87: "f32[32, 320, 8, 8]" = torch.ops.aten.relu.default(add_442);  add_442 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_88: "f32[32, 384, 8, 8]" = torch.ops.aten.convolution.default(cat_12, primals_267, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_443: "i64[]" = torch.ops.aten.add.Tensor(primals_559, 1)
        var_mean_88 = torch.ops.aten.var_mean.correction(convolution_88, [0, 2, 3], correction = 0, keepdim = True)
        getitem_184: "f32[1, 384, 1, 1]" = var_mean_88[0]
        getitem_185: "f32[1, 384, 1, 1]" = var_mean_88[1];  var_mean_88 = None
        add_444: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_184, 0.001)
        rsqrt_88: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_444);  add_444 = None
        sub_88: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_88, getitem_185)
        mul_619: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_88);  sub_88 = None
        squeeze_264: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_185, [0, 2, 3]);  getitem_185 = None
        squeeze_265: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_88, [0, 2, 3]);  rsqrt_88 = None
        mul_620: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_264, 0.1)
        mul_621: "f32[384]" = torch.ops.aten.mul.Tensor(primals_557, 0.9)
        add_445: "f32[384]" = torch.ops.aten.add.Tensor(mul_620, mul_621);  mul_620 = mul_621 = None
        squeeze_266: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_184, [0, 2, 3]);  getitem_184 = None
        mul_622: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_266, 1.0004885197850513);  squeeze_266 = None
        mul_623: "f32[384]" = torch.ops.aten.mul.Tensor(mul_622, 0.1);  mul_622 = None
        mul_624: "f32[384]" = torch.ops.aten.mul.Tensor(primals_558, 0.9)
        add_446: "f32[384]" = torch.ops.aten.add.Tensor(mul_623, mul_624);  mul_623 = mul_624 = None
        unsqueeze_355: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_268, -1)
        unsqueeze_356: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, -1);  unsqueeze_355 = None
        mul_625: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_619, unsqueeze_356);  mul_619 = unsqueeze_356 = None
        unsqueeze_357: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_269, -1);  primals_269 = None
        unsqueeze_358: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, -1);  unsqueeze_357 = None
        add_447: "f32[32, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_625, unsqueeze_358);  mul_625 = unsqueeze_358 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_88: "f32[32, 384, 8, 8]" = torch.ops.aten.relu.default(add_447);  add_447 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_89: "f32[32, 384, 8, 8]" = torch.ops.aten.convolution.default(relu_88, primals_270, None, [1, 1], [0, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_448: "i64[]" = torch.ops.aten.add.Tensor(primals_562, 1)
        var_mean_89 = torch.ops.aten.var_mean.correction(convolution_89, [0, 2, 3], correction = 0, keepdim = True)
        getitem_186: "f32[1, 384, 1, 1]" = var_mean_89[0]
        getitem_187: "f32[1, 384, 1, 1]" = var_mean_89[1];  var_mean_89 = None
        add_449: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_186, 0.001)
        rsqrt_89: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_449);  add_449 = None
        sub_89: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_89, getitem_187)
        mul_626: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_89);  sub_89 = None
        squeeze_267: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_187, [0, 2, 3]);  getitem_187 = None
        squeeze_268: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_89, [0, 2, 3]);  rsqrt_89 = None
        mul_627: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_267, 0.1)
        mul_628: "f32[384]" = torch.ops.aten.mul.Tensor(primals_560, 0.9)
        add_450: "f32[384]" = torch.ops.aten.add.Tensor(mul_627, mul_628);  mul_627 = mul_628 = None
        squeeze_269: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_186, [0, 2, 3]);  getitem_186 = None
        mul_629: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_269, 1.0004885197850513);  squeeze_269 = None
        mul_630: "f32[384]" = torch.ops.aten.mul.Tensor(mul_629, 0.1);  mul_629 = None
        mul_631: "f32[384]" = torch.ops.aten.mul.Tensor(primals_561, 0.9)
        add_451: "f32[384]" = torch.ops.aten.add.Tensor(mul_630, mul_631);  mul_630 = mul_631 = None
        unsqueeze_359: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_271, -1)
        unsqueeze_360: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, -1);  unsqueeze_359 = None
        mul_632: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_626, unsqueeze_360);  mul_626 = unsqueeze_360 = None
        unsqueeze_361: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_272, -1);  primals_272 = None
        unsqueeze_362: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, -1);  unsqueeze_361 = None
        add_452: "f32[32, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_632, unsqueeze_362);  mul_632 = unsqueeze_362 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_89: "f32[32, 384, 8, 8]" = torch.ops.aten.relu.default(add_452);  add_452 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_90: "f32[32, 384, 8, 8]" = torch.ops.aten.convolution.default(relu_88, primals_273, None, [1, 1], [1, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_453: "i64[]" = torch.ops.aten.add.Tensor(primals_565, 1)
        var_mean_90 = torch.ops.aten.var_mean.correction(convolution_90, [0, 2, 3], correction = 0, keepdim = True)
        getitem_188: "f32[1, 384, 1, 1]" = var_mean_90[0]
        getitem_189: "f32[1, 384, 1, 1]" = var_mean_90[1];  var_mean_90 = None
        add_454: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_188, 0.001)
        rsqrt_90: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_454);  add_454 = None
        sub_90: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_90, getitem_189)
        mul_633: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_90);  sub_90 = None
        squeeze_270: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_189, [0, 2, 3]);  getitem_189 = None
        squeeze_271: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_90, [0, 2, 3]);  rsqrt_90 = None
        mul_634: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_270, 0.1)
        mul_635: "f32[384]" = torch.ops.aten.mul.Tensor(primals_563, 0.9)
        add_455: "f32[384]" = torch.ops.aten.add.Tensor(mul_634, mul_635);  mul_634 = mul_635 = None
        squeeze_272: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_188, [0, 2, 3]);  getitem_188 = None
        mul_636: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_272, 1.0004885197850513);  squeeze_272 = None
        mul_637: "f32[384]" = torch.ops.aten.mul.Tensor(mul_636, 0.1);  mul_636 = None
        mul_638: "f32[384]" = torch.ops.aten.mul.Tensor(primals_564, 0.9)
        add_456: "f32[384]" = torch.ops.aten.add.Tensor(mul_637, mul_638);  mul_637 = mul_638 = None
        unsqueeze_363: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_274, -1)
        unsqueeze_364: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, -1);  unsqueeze_363 = None
        mul_639: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_633, unsqueeze_364);  mul_633 = unsqueeze_364 = None
        unsqueeze_365: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_275, -1);  primals_275 = None
        unsqueeze_366: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, -1);  unsqueeze_365 = None
        add_457: "f32[32, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_639, unsqueeze_366);  mul_639 = unsqueeze_366 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_90: "f32[32, 384, 8, 8]" = torch.ops.aten.relu.default(add_457);  add_457 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:346 in _forward, code: branch3x3 = torch.cat(branch3x3, 1)
        cat_13: "f32[32, 768, 8, 8]" = torch.ops.aten.cat.default([relu_89, relu_90], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_91: "f32[32, 448, 8, 8]" = torch.ops.aten.convolution.default(cat_12, primals_276, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_458: "i64[]" = torch.ops.aten.add.Tensor(primals_568, 1)
        var_mean_91 = torch.ops.aten.var_mean.correction(convolution_91, [0, 2, 3], correction = 0, keepdim = True)
        getitem_190: "f32[1, 448, 1, 1]" = var_mean_91[0]
        getitem_191: "f32[1, 448, 1, 1]" = var_mean_91[1];  var_mean_91 = None
        add_459: "f32[1, 448, 1, 1]" = torch.ops.aten.add.Tensor(getitem_190, 0.001)
        rsqrt_91: "f32[1, 448, 1, 1]" = torch.ops.aten.rsqrt.default(add_459);  add_459 = None
        sub_91: "f32[32, 448, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_91, getitem_191)
        mul_640: "f32[32, 448, 8, 8]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_91);  sub_91 = None
        squeeze_273: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_191, [0, 2, 3]);  getitem_191 = None
        squeeze_274: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_91, [0, 2, 3]);  rsqrt_91 = None
        mul_641: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_273, 0.1)
        mul_642: "f32[448]" = torch.ops.aten.mul.Tensor(primals_566, 0.9)
        add_460: "f32[448]" = torch.ops.aten.add.Tensor(mul_641, mul_642);  mul_641 = mul_642 = None
        squeeze_275: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_190, [0, 2, 3]);  getitem_190 = None
        mul_643: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_275, 1.0004885197850513);  squeeze_275 = None
        mul_644: "f32[448]" = torch.ops.aten.mul.Tensor(mul_643, 0.1);  mul_643 = None
        mul_645: "f32[448]" = torch.ops.aten.mul.Tensor(primals_567, 0.9)
        add_461: "f32[448]" = torch.ops.aten.add.Tensor(mul_644, mul_645);  mul_644 = mul_645 = None
        unsqueeze_367: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_277, -1)
        unsqueeze_368: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, -1);  unsqueeze_367 = None
        mul_646: "f32[32, 448, 8, 8]" = torch.ops.aten.mul.Tensor(mul_640, unsqueeze_368);  mul_640 = unsqueeze_368 = None
        unsqueeze_369: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_278, -1);  primals_278 = None
        unsqueeze_370: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, -1);  unsqueeze_369 = None
        add_462: "f32[32, 448, 8, 8]" = torch.ops.aten.add.Tensor(mul_646, unsqueeze_370);  mul_646 = unsqueeze_370 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_91: "f32[32, 448, 8, 8]" = torch.ops.aten.relu.default(add_462);  add_462 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_92: "f32[32, 384, 8, 8]" = torch.ops.aten.convolution.default(relu_91, primals_279, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_463: "i64[]" = torch.ops.aten.add.Tensor(primals_571, 1)
        var_mean_92 = torch.ops.aten.var_mean.correction(convolution_92, [0, 2, 3], correction = 0, keepdim = True)
        getitem_192: "f32[1, 384, 1, 1]" = var_mean_92[0]
        getitem_193: "f32[1, 384, 1, 1]" = var_mean_92[1];  var_mean_92 = None
        add_464: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_192, 0.001)
        rsqrt_92: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_464);  add_464 = None
        sub_92: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_92, getitem_193)
        mul_647: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_92);  sub_92 = None
        squeeze_276: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_193, [0, 2, 3]);  getitem_193 = None
        squeeze_277: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_92, [0, 2, 3]);  rsqrt_92 = None
        mul_648: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_276, 0.1)
        mul_649: "f32[384]" = torch.ops.aten.mul.Tensor(primals_569, 0.9)
        add_465: "f32[384]" = torch.ops.aten.add.Tensor(mul_648, mul_649);  mul_648 = mul_649 = None
        squeeze_278: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_192, [0, 2, 3]);  getitem_192 = None
        mul_650: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_278, 1.0004885197850513);  squeeze_278 = None
        mul_651: "f32[384]" = torch.ops.aten.mul.Tensor(mul_650, 0.1);  mul_650 = None
        mul_652: "f32[384]" = torch.ops.aten.mul.Tensor(primals_570, 0.9)
        add_466: "f32[384]" = torch.ops.aten.add.Tensor(mul_651, mul_652);  mul_651 = mul_652 = None
        unsqueeze_371: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_280, -1)
        unsqueeze_372: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, -1);  unsqueeze_371 = None
        mul_653: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_647, unsqueeze_372);  mul_647 = unsqueeze_372 = None
        unsqueeze_373: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_281, -1);  primals_281 = None
        unsqueeze_374: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, -1);  unsqueeze_373 = None
        add_467: "f32[32, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_653, unsqueeze_374);  mul_653 = unsqueeze_374 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_92: "f32[32, 384, 8, 8]" = torch.ops.aten.relu.default(add_467);  add_467 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_93: "f32[32, 384, 8, 8]" = torch.ops.aten.convolution.default(relu_92, primals_282, None, [1, 1], [0, 1], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_468: "i64[]" = torch.ops.aten.add.Tensor(primals_574, 1)
        var_mean_93 = torch.ops.aten.var_mean.correction(convolution_93, [0, 2, 3], correction = 0, keepdim = True)
        getitem_194: "f32[1, 384, 1, 1]" = var_mean_93[0]
        getitem_195: "f32[1, 384, 1, 1]" = var_mean_93[1];  var_mean_93 = None
        add_469: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_194, 0.001)
        rsqrt_93: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_469);  add_469 = None
        sub_93: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_93, getitem_195)
        mul_654: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_93);  sub_93 = None
        squeeze_279: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_195, [0, 2, 3]);  getitem_195 = None
        squeeze_280: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_93, [0, 2, 3]);  rsqrt_93 = None
        mul_655: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_279, 0.1)
        mul_656: "f32[384]" = torch.ops.aten.mul.Tensor(primals_572, 0.9)
        add_470: "f32[384]" = torch.ops.aten.add.Tensor(mul_655, mul_656);  mul_655 = mul_656 = None
        squeeze_281: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_194, [0, 2, 3]);  getitem_194 = None
        mul_657: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_281, 1.0004885197850513);  squeeze_281 = None
        mul_658: "f32[384]" = torch.ops.aten.mul.Tensor(mul_657, 0.1);  mul_657 = None
        mul_659: "f32[384]" = torch.ops.aten.mul.Tensor(primals_573, 0.9)
        add_471: "f32[384]" = torch.ops.aten.add.Tensor(mul_658, mul_659);  mul_658 = mul_659 = None
        unsqueeze_375: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_283, -1)
        unsqueeze_376: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, -1);  unsqueeze_375 = None
        mul_660: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_654, unsqueeze_376);  mul_654 = unsqueeze_376 = None
        unsqueeze_377: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_284, -1);  primals_284 = None
        unsqueeze_378: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, -1);  unsqueeze_377 = None
        add_472: "f32[32, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_660, unsqueeze_378);  mul_660 = unsqueeze_378 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_93: "f32[32, 384, 8, 8]" = torch.ops.aten.relu.default(add_472);  add_472 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_94: "f32[32, 384, 8, 8]" = torch.ops.aten.convolution.default(relu_92, primals_285, None, [1, 1], [1, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_473: "i64[]" = torch.ops.aten.add.Tensor(primals_577, 1)
        var_mean_94 = torch.ops.aten.var_mean.correction(convolution_94, [0, 2, 3], correction = 0, keepdim = True)
        getitem_196: "f32[1, 384, 1, 1]" = var_mean_94[0]
        getitem_197: "f32[1, 384, 1, 1]" = var_mean_94[1];  var_mean_94 = None
        add_474: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_196, 0.001)
        rsqrt_94: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_474);  add_474 = None
        sub_94: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_94, getitem_197)
        mul_661: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_94);  sub_94 = None
        squeeze_282: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_197, [0, 2, 3]);  getitem_197 = None
        squeeze_283: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_94, [0, 2, 3]);  rsqrt_94 = None
        mul_662: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_282, 0.1)
        mul_663: "f32[384]" = torch.ops.aten.mul.Tensor(primals_575, 0.9)
        add_475: "f32[384]" = torch.ops.aten.add.Tensor(mul_662, mul_663);  mul_662 = mul_663 = None
        squeeze_284: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_196, [0, 2, 3]);  getitem_196 = None
        mul_664: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_284, 1.0004885197850513);  squeeze_284 = None
        mul_665: "f32[384]" = torch.ops.aten.mul.Tensor(mul_664, 0.1);  mul_664 = None
        mul_666: "f32[384]" = torch.ops.aten.mul.Tensor(primals_576, 0.9)
        add_476: "f32[384]" = torch.ops.aten.add.Tensor(mul_665, mul_666);  mul_665 = mul_666 = None
        unsqueeze_379: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_286, -1)
        unsqueeze_380: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, -1);  unsqueeze_379 = None
        mul_667: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_661, unsqueeze_380);  mul_661 = unsqueeze_380 = None
        unsqueeze_381: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_287, -1);  primals_287 = None
        unsqueeze_382: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, -1);  unsqueeze_381 = None
        add_477: "f32[32, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_667, unsqueeze_382);  mul_667 = unsqueeze_382 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_94: "f32[32, 384, 8, 8]" = torch.ops.aten.relu.default(add_477);  add_477 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:354 in _forward, code: branch3x3dbl = torch.cat(branch3x3dbl, 1)
        cat_14: "f32[32, 768, 8, 8]" = torch.ops.aten.cat.default([relu_93, relu_94], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:356 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_9: "f32[32, 2048, 8, 8]" = torch.ops.aten.avg_pool2d.default(cat_12, [3, 3], [1, 1], [1, 1])
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:405 in forward, code: x = self.conv(x)
        convolution_95: "f32[32, 192, 8, 8]" = torch.ops.aten.convolution.default(avg_pool2d_9, primals_288, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        add_478: "i64[]" = torch.ops.aten.add.Tensor(primals_580, 1)
        var_mean_95 = torch.ops.aten.var_mean.correction(convolution_95, [0, 2, 3], correction = 0, keepdim = True)
        getitem_198: "f32[1, 192, 1, 1]" = var_mean_95[0]
        getitem_199: "f32[1, 192, 1, 1]" = var_mean_95[1];  var_mean_95 = None
        add_479: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_198, 0.001)
        rsqrt_95: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_479);  add_479 = None
        sub_95: "f32[32, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_95, getitem_199)
        mul_668: "f32[32, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_95);  sub_95 = None
        squeeze_285: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_199, [0, 2, 3]);  getitem_199 = None
        squeeze_286: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_95, [0, 2, 3]);  rsqrt_95 = None
        mul_669: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_285, 0.1)
        mul_670: "f32[192]" = torch.ops.aten.mul.Tensor(primals_578, 0.9)
        add_480: "f32[192]" = torch.ops.aten.add.Tensor(mul_669, mul_670);  mul_669 = mul_670 = None
        squeeze_287: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_198, [0, 2, 3]);  getitem_198 = None
        mul_671: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_287, 1.0004885197850513);  squeeze_287 = None
        mul_672: "f32[192]" = torch.ops.aten.mul.Tensor(mul_671, 0.1);  mul_671 = None
        mul_673: "f32[192]" = torch.ops.aten.mul.Tensor(primals_579, 0.9)
        add_481: "f32[192]" = torch.ops.aten.add.Tensor(mul_672, mul_673);  mul_672 = mul_673 = None
        unsqueeze_383: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_289, -1)
        unsqueeze_384: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, -1);  unsqueeze_383 = None
        mul_674: "f32[32, 192, 8, 8]" = torch.ops.aten.mul.Tensor(mul_668, unsqueeze_384);  mul_668 = unsqueeze_384 = None
        unsqueeze_385: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_290, -1);  primals_290 = None
        unsqueeze_386: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, -1);  unsqueeze_385 = None
        add_482: "f32[32, 192, 8, 8]" = torch.ops.aten.add.Tensor(mul_674, unsqueeze_386);  mul_674 = unsqueeze_386 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        relu_95: "f32[32, 192, 8, 8]" = torch.ops.aten.relu.default(add_482);  add_482 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:364 in forward, code: return torch.cat(outputs, 1)
        cat_15: "f32[32, 2048, 8, 8]" = torch.ops.aten.cat.default([relu_87, cat_13, cat_14, relu_95], 1);  cat_13 = cat_14 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:147 in _forward, code: x = self.avgpool(x)
        mean_1: "f32[32, 2048, 1, 1]" = torch.ops.aten.mean.dim(cat_15, [-1, -2], True);  cat_15 = None
        
        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[32, 2048, 1, 1]" = torch.ops.prims.inductor_random.default([32, 2048, 1, 1], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:149 in _forward, code: x = self.dropout(x)
        gt: "b8[32, 2048, 1, 1]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.5);  inductor_random_default = None
        mul_675: "f32[32, 2048, 1, 1]" = torch.ops.aten.mul.Tensor(gt, mean_1);  mean_1 = None
        mul_676: "f32[32, 2048, 1, 1]" = torch.ops.aten.mul.Tensor(mul_675, 2.0);  mul_675 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:151 in _forward, code: x = torch.flatten(x, 1)
        view_1: "f32[32, 2048]" = torch.ops.aten.view.default(mul_676, [32, 2048]);  mul_676 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:153 in _forward, code: x = self.fc(x)
        permute_1: "f32[2048, 1000]" = torch.ops.aten.permute.default(primals_291, [1, 0]);  primals_291 = None
        addmm_1: "f32[32, 1000]" = torch.ops.aten.addmm.default(primals_292, view_1, permute_1);  primals_292 = None
        permute_2: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_194: "f32[32, 192, 8, 8]" = torch.ops.aten.alias.default(relu_95);  relu_95 = None
        alias_195: "f32[32, 192, 8, 8]" = torch.ops.aten.alias.default(alias_194);  alias_194 = None
        le: "b8[32, 192, 8, 8]" = torch.ops.aten.le.Scalar(alias_195, 0);  alias_195 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_387: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_285, 0);  squeeze_285 = None
        unsqueeze_388: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 2);  unsqueeze_387 = None
        unsqueeze_389: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 3);  unsqueeze_388 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_198: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(relu_94);  relu_94 = None
        alias_199: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(alias_198);  alias_198 = None
        le_1: "b8[32, 384, 8, 8]" = torch.ops.aten.le.Scalar(alias_199, 0);  alias_199 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_399: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_282, 0);  squeeze_282 = None
        unsqueeze_400: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
        unsqueeze_401: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 3);  unsqueeze_400 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_202: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(relu_93);  relu_93 = None
        alias_203: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(alias_202);  alias_202 = None
        le_2: "b8[32, 384, 8, 8]" = torch.ops.aten.le.Scalar(alias_203, 0);  alias_203 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_411: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_279, 0);  squeeze_279 = None
        unsqueeze_412: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
        unsqueeze_413: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 3);  unsqueeze_412 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_423: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_276, 0);  squeeze_276 = None
        unsqueeze_424: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
        unsqueeze_425: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 3);  unsqueeze_424 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_435: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_273, 0);  squeeze_273 = None
        unsqueeze_436: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 2);  unsqueeze_435 = None
        unsqueeze_437: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 3);  unsqueeze_436 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_214: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(relu_90);  relu_90 = None
        alias_215: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(alias_214);  alias_214 = None
        le_5: "b8[32, 384, 8, 8]" = torch.ops.aten.le.Scalar(alias_215, 0);  alias_215 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_447: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_270, 0);  squeeze_270 = None
        unsqueeze_448: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 2);  unsqueeze_447 = None
        unsqueeze_449: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 3);  unsqueeze_448 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_218: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(relu_89);  relu_89 = None
        alias_219: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(alias_218);  alias_218 = None
        le_6: "b8[32, 384, 8, 8]" = torch.ops.aten.le.Scalar(alias_219, 0);  alias_219 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_459: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_267, 0);  squeeze_267 = None
        unsqueeze_460: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 2);  unsqueeze_459 = None
        unsqueeze_461: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 3);  unsqueeze_460 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_471: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_264, 0);  squeeze_264 = None
        unsqueeze_472: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 2);  unsqueeze_471 = None
        unsqueeze_473: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 3);  unsqueeze_472 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_226: "f32[32, 320, 8, 8]" = torch.ops.aten.alias.default(relu_87);  relu_87 = None
        alias_227: "f32[32, 320, 8, 8]" = torch.ops.aten.alias.default(alias_226);  alias_226 = None
        le_8: "b8[32, 320, 8, 8]" = torch.ops.aten.le.Scalar(alias_227, 0);  alias_227 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_483: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(squeeze_261, 0);  squeeze_261 = None
        unsqueeze_484: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 2);  unsqueeze_483 = None
        unsqueeze_485: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 3);  unsqueeze_484 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_230: "f32[32, 192, 8, 8]" = torch.ops.aten.alias.default(relu_86);  relu_86 = None
        alias_231: "f32[32, 192, 8, 8]" = torch.ops.aten.alias.default(alias_230);  alias_230 = None
        le_9: "b8[32, 192, 8, 8]" = torch.ops.aten.le.Scalar(alias_231, 0);  alias_231 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_495: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_258, 0);  squeeze_258 = None
        unsqueeze_496: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 2);  unsqueeze_495 = None
        unsqueeze_497: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 3);  unsqueeze_496 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_234: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(relu_85);  relu_85 = None
        alias_235: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(alias_234);  alias_234 = None
        le_10: "b8[32, 384, 8, 8]" = torch.ops.aten.le.Scalar(alias_235, 0);  alias_235 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_507: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_255, 0);  squeeze_255 = None
        unsqueeze_508: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 2);  unsqueeze_507 = None
        unsqueeze_509: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 3);  unsqueeze_508 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_238: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(relu_84);  relu_84 = None
        alias_239: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(alias_238);  alias_238 = None
        le_11: "b8[32, 384, 8, 8]" = torch.ops.aten.le.Scalar(alias_239, 0);  alias_239 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_519: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_252, 0);  squeeze_252 = None
        unsqueeze_520: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 2);  unsqueeze_519 = None
        unsqueeze_521: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 3);  unsqueeze_520 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_531: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_249, 0);  squeeze_249 = None
        unsqueeze_532: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 2);  unsqueeze_531 = None
        unsqueeze_533: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 3);  unsqueeze_532 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_543: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_246, 0);  squeeze_246 = None
        unsqueeze_544: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 2);  unsqueeze_543 = None
        unsqueeze_545: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 3);  unsqueeze_544 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_250: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(relu_81);  relu_81 = None
        alias_251: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(alias_250);  alias_250 = None
        le_14: "b8[32, 384, 8, 8]" = torch.ops.aten.le.Scalar(alias_251, 0);  alias_251 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_555: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_243, 0);  squeeze_243 = None
        unsqueeze_556: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 2);  unsqueeze_555 = None
        unsqueeze_557: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 3);  unsqueeze_556 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_254: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(relu_80);  relu_80 = None
        alias_255: "f32[32, 384, 8, 8]" = torch.ops.aten.alias.default(alias_254);  alias_254 = None
        le_15: "b8[32, 384, 8, 8]" = torch.ops.aten.le.Scalar(alias_255, 0);  alias_255 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_567: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_240, 0);  squeeze_240 = None
        unsqueeze_568: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 2);  unsqueeze_567 = None
        unsqueeze_569: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 3);  unsqueeze_568 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_579: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_237, 0);  squeeze_237 = None
        unsqueeze_580: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 2);  unsqueeze_579 = None
        unsqueeze_581: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 3);  unsqueeze_580 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_262: "f32[32, 320, 8, 8]" = torch.ops.aten.alias.default(relu_78);  relu_78 = None
        alias_263: "f32[32, 320, 8, 8]" = torch.ops.aten.alias.default(alias_262);  alias_262 = None
        le_17: "b8[32, 320, 8, 8]" = torch.ops.aten.le.Scalar(alias_263, 0);  alias_263 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_591: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(squeeze_234, 0);  squeeze_234 = None
        unsqueeze_592: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 2);  unsqueeze_591 = None
        unsqueeze_593: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 3);  unsqueeze_592 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_266: "f32[32, 192, 8, 8]" = torch.ops.aten.alias.default(relu_77);  relu_77 = None
        alias_267: "f32[32, 192, 8, 8]" = torch.ops.aten.alias.default(alias_266);  alias_266 = None
        le_18: "b8[32, 192, 8, 8]" = torch.ops.aten.le.Scalar(alias_267, 0);  alias_267 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_603: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_231, 0);  squeeze_231 = None
        unsqueeze_604: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 2);  unsqueeze_603 = None
        unsqueeze_605: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 3);  unsqueeze_604 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_615: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_228, 0);  squeeze_228 = None
        unsqueeze_616: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 2);  unsqueeze_615 = None
        unsqueeze_617: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 3);  unsqueeze_616 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_627: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_225, 0);  squeeze_225 = None
        unsqueeze_628: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 2);  unsqueeze_627 = None
        unsqueeze_629: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 3);  unsqueeze_628 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_639: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_222, 0);  squeeze_222 = None
        unsqueeze_640: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 2);  unsqueeze_639 = None
        unsqueeze_641: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 3);  unsqueeze_640 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_282: "f32[32, 320, 8, 8]" = torch.ops.aten.alias.default(relu_73);  relu_73 = None
        alias_283: "f32[32, 320, 8, 8]" = torch.ops.aten.alias.default(alias_282);  alias_282 = None
        le_22: "b8[32, 320, 8, 8]" = torch.ops.aten.le.Scalar(alias_283, 0);  alias_283 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_651: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(squeeze_219, 0);  squeeze_219 = None
        unsqueeze_652: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 2);  unsqueeze_651 = None
        unsqueeze_653: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 3);  unsqueeze_652 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_663: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_216, 0);  squeeze_216 = None
        unsqueeze_664: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 2);  unsqueeze_663 = None
        unsqueeze_665: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 3);  unsqueeze_664 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:393 in forward, code: x = self.fc(x)
        permute_6: "f32[1000, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_290: "f32[32, 768, 1, 1]" = torch.ops.aten.alias.default(relu_71);  relu_71 = None
        alias_291: "f32[32, 768, 1, 1]" = torch.ops.aten.alias.default(alias_290);  alias_290 = None
        le_24: "b8[32, 768, 1, 1]" = torch.ops.aten.le.Scalar(alias_291, 0);  alias_291 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_675: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_213, 0);  squeeze_213 = None
        unsqueeze_676: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 2);  unsqueeze_675 = None
        unsqueeze_677: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 3);  unsqueeze_676 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_687: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_210, 0);  squeeze_210 = None
        unsqueeze_688: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 2);  unsqueeze_687 = None
        unsqueeze_689: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 3);  unsqueeze_688 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_298: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_69);  relu_69 = None
        alias_299: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_298);  alias_298 = None
        le_26: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_299, 0);  alias_299 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_699: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_207, 0);  squeeze_207 = None
        unsqueeze_700: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 2);  unsqueeze_699 = None
        unsqueeze_701: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 3);  unsqueeze_700 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_302: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_68);  relu_68 = None
        alias_303: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_302);  alias_302 = None
        le_27: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_303, 0);  alias_303 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_711: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_204, 0);  squeeze_204 = None
        unsqueeze_712: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 2);  unsqueeze_711 = None
        unsqueeze_713: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 3);  unsqueeze_712 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_723: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_201, 0);  squeeze_201 = None
        unsqueeze_724: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 2);  unsqueeze_723 = None
        unsqueeze_725: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 3);  unsqueeze_724 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_735: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_198, 0);  squeeze_198 = None
        unsqueeze_736: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 2);  unsqueeze_735 = None
        unsqueeze_737: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 3);  unsqueeze_736 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_747: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_195, 0);  squeeze_195 = None
        unsqueeze_748: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 2);  unsqueeze_747 = None
        unsqueeze_749: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 3);  unsqueeze_748 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_759: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_192, 0);  squeeze_192 = None
        unsqueeze_760: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 2);  unsqueeze_759 = None
        unsqueeze_761: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 3);  unsqueeze_760 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_322: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_63);  relu_63 = None
        alias_323: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_322);  alias_322 = None
        le_32: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_323, 0);  alias_323 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_771: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_189, 0);  squeeze_189 = None
        unsqueeze_772: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 2);  unsqueeze_771 = None
        unsqueeze_773: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 3);  unsqueeze_772 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_783: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_186, 0);  squeeze_186 = None
        unsqueeze_784: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 2);  unsqueeze_783 = None
        unsqueeze_785: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 3);  unsqueeze_784 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_795: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
        unsqueeze_796: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 2);  unsqueeze_795 = None
        unsqueeze_797: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 3);  unsqueeze_796 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_334: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_60);  relu_60 = None
        alias_335: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_334);  alias_334 = None
        le_35: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_335, 0);  alias_335 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_807: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
        unsqueeze_808: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 2);  unsqueeze_807 = None
        unsqueeze_809: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 3);  unsqueeze_808 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_338: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_59);  relu_59 = None
        alias_339: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_338);  alias_338 = None
        le_36: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_339, 0);  alias_339 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_819: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
        unsqueeze_820: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 2);  unsqueeze_819 = None
        unsqueeze_821: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 3);  unsqueeze_820 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_342: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_58);  relu_58 = None
        alias_343: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_342);  alias_342 = None
        le_37: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_343, 0);  alias_343 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_831: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
        unsqueeze_832: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 2);  unsqueeze_831 = None
        unsqueeze_833: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 3);  unsqueeze_832 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_843: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
        unsqueeze_844: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_843, 2);  unsqueeze_843 = None
        unsqueeze_845: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 3);  unsqueeze_844 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_855: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
        unsqueeze_856: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_855, 2);  unsqueeze_855 = None
        unsqueeze_857: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 3);  unsqueeze_856 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_867: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
        unsqueeze_868: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_867, 2);  unsqueeze_867 = None
        unsqueeze_869: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 3);  unsqueeze_868 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_879: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
        unsqueeze_880: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_879, 2);  unsqueeze_879 = None
        unsqueeze_881: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 3);  unsqueeze_880 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_362: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_53);  relu_53 = None
        alias_363: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_362);  alias_362 = None
        le_42: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_363, 0);  alias_363 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_891: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
        unsqueeze_892: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 2);  unsqueeze_891 = None
        unsqueeze_893: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 3);  unsqueeze_892 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_903: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
        unsqueeze_904: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 2);  unsqueeze_903 = None
        unsqueeze_905: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 3);  unsqueeze_904 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_915: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
        unsqueeze_916: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_915, 2);  unsqueeze_915 = None
        unsqueeze_917: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 3);  unsqueeze_916 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_374: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_50);  relu_50 = None
        alias_375: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_374);  alias_374 = None
        le_45: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_375, 0);  alias_375 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_927: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
        unsqueeze_928: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_927, 2);  unsqueeze_927 = None
        unsqueeze_929: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, 3);  unsqueeze_928 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_378: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_49);  relu_49 = None
        alias_379: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_378);  alias_378 = None
        le_46: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_379, 0);  alias_379 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_939: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
        unsqueeze_940: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_939, 2);  unsqueeze_939 = None
        unsqueeze_941: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, 3);  unsqueeze_940 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_382: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_48);  relu_48 = None
        alias_383: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_382);  alias_382 = None
        le_47: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_383, 0);  alias_383 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_951: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
        unsqueeze_952: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_951, 2);  unsqueeze_951 = None
        unsqueeze_953: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, 3);  unsqueeze_952 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_963: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
        unsqueeze_964: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_963, 2);  unsqueeze_963 = None
        unsqueeze_965: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, 3);  unsqueeze_964 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_975: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
        unsqueeze_976: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_975, 2);  unsqueeze_975 = None
        unsqueeze_977: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, 3);  unsqueeze_976 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_987: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
        unsqueeze_988: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_987, 2);  unsqueeze_987 = None
        unsqueeze_989: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, 3);  unsqueeze_988 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_999: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
        unsqueeze_1000: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_999, 2);  unsqueeze_999 = None
        unsqueeze_1001: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, 3);  unsqueeze_1000 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_402: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_43);  relu_43 = None
        alias_403: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_402);  alias_402 = None
        le_52: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_403, 0);  alias_403 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1011: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
        unsqueeze_1012: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1011, 2);  unsqueeze_1011 = None
        unsqueeze_1013: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, 3);  unsqueeze_1012 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1023: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
        unsqueeze_1024: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1023, 2);  unsqueeze_1023 = None
        unsqueeze_1025: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, 3);  unsqueeze_1024 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1035: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
        unsqueeze_1036: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1035, 2);  unsqueeze_1035 = None
        unsqueeze_1037: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, 3);  unsqueeze_1036 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_414: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_40);  relu_40 = None
        alias_415: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_414);  alias_414 = None
        le_55: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_415, 0);  alias_415 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1047: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
        unsqueeze_1048: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1047, 2);  unsqueeze_1047 = None
        unsqueeze_1049: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, 3);  unsqueeze_1048 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_418: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_39);  relu_39 = None
        alias_419: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_418);  alias_418 = None
        le_56: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_419, 0);  alias_419 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1059: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
        unsqueeze_1060: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1059, 2);  unsqueeze_1059 = None
        unsqueeze_1061: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, 3);  unsqueeze_1060 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_422: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_38);  relu_38 = None
        alias_423: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_422);  alias_422 = None
        le_57: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_423, 0);  alias_423 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1071: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
        unsqueeze_1072: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1071, 2);  unsqueeze_1071 = None
        unsqueeze_1073: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, 3);  unsqueeze_1072 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1083: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
        unsqueeze_1084: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1083, 2);  unsqueeze_1083 = None
        unsqueeze_1085: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, 3);  unsqueeze_1084 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1095: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
        unsqueeze_1096: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1095, 2);  unsqueeze_1095 = None
        unsqueeze_1097: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, 3);  unsqueeze_1096 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1107: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
        unsqueeze_1108: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1107, 2);  unsqueeze_1107 = None
        unsqueeze_1109: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, 3);  unsqueeze_1108 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1119: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
        unsqueeze_1120: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1119, 2);  unsqueeze_1119 = None
        unsqueeze_1121: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, 3);  unsqueeze_1120 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_442: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_33);  relu_33 = None
        alias_443: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_442);  alias_442 = None
        le_62: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_443, 0);  alias_443 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1131: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
        unsqueeze_1132: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1131, 2);  unsqueeze_1131 = None
        unsqueeze_1133: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, 3);  unsqueeze_1132 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1143: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
        unsqueeze_1144: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1143, 2);  unsqueeze_1143 = None
        unsqueeze_1145: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, 3);  unsqueeze_1144 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1155: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
        unsqueeze_1156: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1155, 2);  unsqueeze_1155 = None
        unsqueeze_1157: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, 3);  unsqueeze_1156 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_454: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(relu_30);  relu_30 = None
        alias_455: "f32[32, 192, 17, 17]" = torch.ops.aten.alias.default(alias_454);  alias_454 = None
        le_65: "b8[32, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_455, 0);  alias_455 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1167: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
        unsqueeze_1168: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1167, 2);  unsqueeze_1167 = None
        unsqueeze_1169: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, 3);  unsqueeze_1168 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_458: "f32[32, 96, 17, 17]" = torch.ops.aten.alias.default(relu_29);  relu_29 = None
        alias_459: "f32[32, 96, 17, 17]" = torch.ops.aten.alias.default(alias_458);  alias_458 = None
        le_66: "b8[32, 96, 17, 17]" = torch.ops.aten.le.Scalar(alias_459, 0);  alias_459 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1179: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
        unsqueeze_1180: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1179, 2);  unsqueeze_1179 = None
        unsqueeze_1181: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1180, 3);  unsqueeze_1180 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1191: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_1192: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1191, 2);  unsqueeze_1191 = None
        unsqueeze_1193: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1192, 3);  unsqueeze_1192 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1203: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
        unsqueeze_1204: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1203, 2);  unsqueeze_1203 = None
        unsqueeze_1205: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, 3);  unsqueeze_1204 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_470: "f32[32, 384, 17, 17]" = torch.ops.aten.alias.default(relu_26);  relu_26 = None
        alias_471: "f32[32, 384, 17, 17]" = torch.ops.aten.alias.default(alias_470);  alias_470 = None
        le_69: "b8[32, 384, 17, 17]" = torch.ops.aten.le.Scalar(alias_471, 0);  alias_471 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1215: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
        unsqueeze_1216: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1215, 2);  unsqueeze_1215 = None
        unsqueeze_1217: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1216, 3);  unsqueeze_1216 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_474: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
        alias_475: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(alias_474);  alias_474 = None
        le_70: "b8[32, 64, 35, 35]" = torch.ops.aten.le.Scalar(alias_475, 0);  alias_475 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1227: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
        unsqueeze_1228: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1227, 2);  unsqueeze_1227 = None
        unsqueeze_1229: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, 3);  unsqueeze_1228 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_478: "f32[32, 96, 35, 35]" = torch.ops.aten.alias.default(relu_24);  relu_24 = None
        alias_479: "f32[32, 96, 35, 35]" = torch.ops.aten.alias.default(alias_478);  alias_478 = None
        le_71: "b8[32, 96, 35, 35]" = torch.ops.aten.le.Scalar(alias_479, 0);  alias_479 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1239: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_1240: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1239, 2);  unsqueeze_1239 = None
        unsqueeze_1241: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1240, 3);  unsqueeze_1240 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1251: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
        unsqueeze_1252: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1251, 2);  unsqueeze_1251 = None
        unsqueeze_1253: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1252, 3);  unsqueeze_1252 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1263: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_1264: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1263, 2);  unsqueeze_1263 = None
        unsqueeze_1265: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1264, 3);  unsqueeze_1264 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_490: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
        alias_491: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(alias_490);  alias_490 = None
        le_74: "b8[32, 64, 35, 35]" = torch.ops.aten.le.Scalar(alias_491, 0);  alias_491 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1275: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
        unsqueeze_1276: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1275, 2);  unsqueeze_1275 = None
        unsqueeze_1277: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1276, 3);  unsqueeze_1276 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1287: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
        unsqueeze_1288: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1287, 2);  unsqueeze_1287 = None
        unsqueeze_1289: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1288, 3);  unsqueeze_1288 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_498: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
        alias_499: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(alias_498);  alias_498 = None
        le_76: "b8[32, 64, 35, 35]" = torch.ops.aten.le.Scalar(alias_499, 0);  alias_499 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1299: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_1300: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1299, 2);  unsqueeze_1299 = None
        unsqueeze_1301: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1300, 3);  unsqueeze_1300 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_502: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
        alias_503: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(alias_502);  alias_502 = None
        le_77: "b8[32, 64, 35, 35]" = torch.ops.aten.le.Scalar(alias_503, 0);  alias_503 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1311: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_1312: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1311, 2);  unsqueeze_1311 = None
        unsqueeze_1313: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1312, 3);  unsqueeze_1312 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_506: "f32[32, 96, 35, 35]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
        alias_507: "f32[32, 96, 35, 35]" = torch.ops.aten.alias.default(alias_506);  alias_506 = None
        le_78: "b8[32, 96, 35, 35]" = torch.ops.aten.le.Scalar(alias_507, 0);  alias_507 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1323: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_1324: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1323, 2);  unsqueeze_1323 = None
        unsqueeze_1325: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1324, 3);  unsqueeze_1324 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1335: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_1336: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1335, 2);  unsqueeze_1335 = None
        unsqueeze_1337: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1336, 3);  unsqueeze_1336 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1347: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_1348: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1347, 2);  unsqueeze_1347 = None
        unsqueeze_1349: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1348, 3);  unsqueeze_1348 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_518: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
        alias_519: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(alias_518);  alias_518 = None
        le_81: "b8[32, 64, 35, 35]" = torch.ops.aten.le.Scalar(alias_519, 0);  alias_519 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1359: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_1360: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1359, 2);  unsqueeze_1359 = None
        unsqueeze_1361: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1360, 3);  unsqueeze_1360 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1371: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_1372: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1371, 2);  unsqueeze_1371 = None
        unsqueeze_1373: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1372, 3);  unsqueeze_1372 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_526: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
        alias_527: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(alias_526);  alias_526 = None
        le_83: "b8[32, 64, 35, 35]" = torch.ops.aten.le.Scalar(alias_527, 0);  alias_527 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1383: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_1384: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1383, 2);  unsqueeze_1383 = None
        unsqueeze_1385: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1384, 3);  unsqueeze_1384 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_530: "f32[32, 32, 35, 35]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
        alias_531: "f32[32, 32, 35, 35]" = torch.ops.aten.alias.default(alias_530);  alias_530 = None
        le_84: "b8[32, 32, 35, 35]" = torch.ops.aten.le.Scalar(alias_531, 0);  alias_531 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1395: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_1396: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1395, 2);  unsqueeze_1395 = None
        unsqueeze_1397: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1396, 3);  unsqueeze_1396 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_534: "f32[32, 96, 35, 35]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
        alias_535: "f32[32, 96, 35, 35]" = torch.ops.aten.alias.default(alias_534);  alias_534 = None
        le_85: "b8[32, 96, 35, 35]" = torch.ops.aten.le.Scalar(alias_535, 0);  alias_535 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1407: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_1408: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1407, 2);  unsqueeze_1407 = None
        unsqueeze_1409: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1408, 3);  unsqueeze_1408 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1419: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_1420: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1419, 2);  unsqueeze_1419 = None
        unsqueeze_1421: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1420, 3);  unsqueeze_1420 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1431: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_1432: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1431, 2);  unsqueeze_1431 = None
        unsqueeze_1433: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1432, 3);  unsqueeze_1432 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_546: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
        alias_547: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(alias_546);  alias_546 = None
        le_88: "b8[32, 64, 35, 35]" = torch.ops.aten.le.Scalar(alias_547, 0);  alias_547 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1443: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_1444: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1443, 2);  unsqueeze_1443 = None
        unsqueeze_1445: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1444, 3);  unsqueeze_1444 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1455: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_1456: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1455, 2);  unsqueeze_1455 = None
        unsqueeze_1457: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1456, 3);  unsqueeze_1456 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:407 in forward, code: return F.relu(x, inplace=True)
        alias_554: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
        alias_555: "f32[32, 64, 35, 35]" = torch.ops.aten.alias.default(alias_554);  alias_554 = None
        le_90: "b8[32, 64, 35, 35]" = torch.ops.aten.le.Scalar(alias_555, 0);  alias_555 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1467: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_1468: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1467, 2);  unsqueeze_1467 = None
        unsqueeze_1469: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1468, 3);  unsqueeze_1468 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1479: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_1480: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1479, 2);  unsqueeze_1479 = None
        unsqueeze_1481: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1480, 3);  unsqueeze_1480 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1491: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_1492: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1491, 2);  unsqueeze_1491 = None
        unsqueeze_1493: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1492, 3);  unsqueeze_1492 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1503: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_1504: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1503, 2);  unsqueeze_1503 = None
        unsqueeze_1505: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1504, 3);  unsqueeze_1504 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1515: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_1516: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1515, 2);  unsqueeze_1515 = None
        unsqueeze_1517: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1516, 3);  unsqueeze_1516 = None
        
        # File: /home/zhang402/miniconda3/envs/myenv/lib/python3.10/site-packages/torchvision/models/inception.py:406 in forward, code: x = self.bn(x)
        unsqueeze_1527: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_1528: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1527, 2);  unsqueeze_1527 = None
        unsqueeze_1529: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1528, 3);  unsqueeze_1528 = None
        
        # No stacktrace found for following nodes
        copy_: "f32[32]" = torch.ops.aten.copy_.default(primals_293, add_5);  primals_293 = add_5 = None
        copy__1: "f32[32]" = torch.ops.aten.copy_.default(primals_294, add_6);  primals_294 = add_6 = None
        copy__2: "i64[]" = torch.ops.aten.copy_.default(primals_295, add_3);  primals_295 = add_3 = None
        copy__3: "f32[32]" = torch.ops.aten.copy_.default(primals_296, add_10);  primals_296 = add_10 = None
        copy__4: "f32[32]" = torch.ops.aten.copy_.default(primals_297, add_11);  primals_297 = add_11 = None
        copy__5: "i64[]" = torch.ops.aten.copy_.default(primals_298, add_8);  primals_298 = add_8 = None
        copy__6: "f32[64]" = torch.ops.aten.copy_.default(primals_299, add_15);  primals_299 = add_15 = None
        copy__7: "f32[64]" = torch.ops.aten.copy_.default(primals_300, add_16);  primals_300 = add_16 = None
        copy__8: "i64[]" = torch.ops.aten.copy_.default(primals_301, add_13);  primals_301 = add_13 = None
        copy__9: "f32[80]" = torch.ops.aten.copy_.default(primals_302, add_20);  primals_302 = add_20 = None
        copy__10: "f32[80]" = torch.ops.aten.copy_.default(primals_303, add_21);  primals_303 = add_21 = None
        copy__11: "i64[]" = torch.ops.aten.copy_.default(primals_304, add_18);  primals_304 = add_18 = None
        copy__12: "f32[192]" = torch.ops.aten.copy_.default(primals_305, add_25);  primals_305 = add_25 = None
        copy__13: "f32[192]" = torch.ops.aten.copy_.default(primals_306, add_26);  primals_306 = add_26 = None
        copy__14: "i64[]" = torch.ops.aten.copy_.default(primals_307, add_23);  primals_307 = add_23 = None
        copy__15: "f32[64]" = torch.ops.aten.copy_.default(primals_308, add_30);  primals_308 = add_30 = None
        copy__16: "f32[64]" = torch.ops.aten.copy_.default(primals_309, add_31);  primals_309 = add_31 = None
        copy__17: "i64[]" = torch.ops.aten.copy_.default(primals_310, add_28);  primals_310 = add_28 = None
        copy__18: "f32[48]" = torch.ops.aten.copy_.default(primals_311, add_35);  primals_311 = add_35 = None
        copy__19: "f32[48]" = torch.ops.aten.copy_.default(primals_312, add_36);  primals_312 = add_36 = None
        copy__20: "i64[]" = torch.ops.aten.copy_.default(primals_313, add_33);  primals_313 = add_33 = None
        copy__21: "f32[64]" = torch.ops.aten.copy_.default(primals_314, add_40);  primals_314 = add_40 = None
        copy__22: "f32[64]" = torch.ops.aten.copy_.default(primals_315, add_41);  primals_315 = add_41 = None
        copy__23: "i64[]" = torch.ops.aten.copy_.default(primals_316, add_38);  primals_316 = add_38 = None
        copy__24: "f32[64]" = torch.ops.aten.copy_.default(primals_317, add_45);  primals_317 = add_45 = None
        copy__25: "f32[64]" = torch.ops.aten.copy_.default(primals_318, add_46);  primals_318 = add_46 = None
        copy__26: "i64[]" = torch.ops.aten.copy_.default(primals_319, add_43);  primals_319 = add_43 = None
        copy__27: "f32[96]" = torch.ops.aten.copy_.default(primals_320, add_50);  primals_320 = add_50 = None
        copy__28: "f32[96]" = torch.ops.aten.copy_.default(primals_321, add_51);  primals_321 = add_51 = None
        copy__29: "i64[]" = torch.ops.aten.copy_.default(primals_322, add_48);  primals_322 = add_48 = None
        copy__30: "f32[96]" = torch.ops.aten.copy_.default(primals_323, add_55);  primals_323 = add_55 = None
        copy__31: "f32[96]" = torch.ops.aten.copy_.default(primals_324, add_56);  primals_324 = add_56 = None
        copy__32: "i64[]" = torch.ops.aten.copy_.default(primals_325, add_53);  primals_325 = add_53 = None
        copy__33: "f32[32]" = torch.ops.aten.copy_.default(primals_326, add_60);  primals_326 = add_60 = None
        copy__34: "f32[32]" = torch.ops.aten.copy_.default(primals_327, add_61);  primals_327 = add_61 = None
        copy__35: "i64[]" = torch.ops.aten.copy_.default(primals_328, add_58);  primals_328 = add_58 = None
        copy__36: "f32[64]" = torch.ops.aten.copy_.default(primals_329, add_65);  primals_329 = add_65 = None
        copy__37: "f32[64]" = torch.ops.aten.copy_.default(primals_330, add_66);  primals_330 = add_66 = None
        copy__38: "i64[]" = torch.ops.aten.copy_.default(primals_331, add_63);  primals_331 = add_63 = None
        copy__39: "f32[48]" = torch.ops.aten.copy_.default(primals_332, add_70);  primals_332 = add_70 = None
        copy__40: "f32[48]" = torch.ops.aten.copy_.default(primals_333, add_71);  primals_333 = add_71 = None
        copy__41: "i64[]" = torch.ops.aten.copy_.default(primals_334, add_68);  primals_334 = add_68 = None
        copy__42: "f32[64]" = torch.ops.aten.copy_.default(primals_335, add_75);  primals_335 = add_75 = None
        copy__43: "f32[64]" = torch.ops.aten.copy_.default(primals_336, add_76);  primals_336 = add_76 = None
        copy__44: "i64[]" = torch.ops.aten.copy_.default(primals_337, add_73);  primals_337 = add_73 = None
        copy__45: "f32[64]" = torch.ops.aten.copy_.default(primals_338, add_80);  primals_338 = add_80 = None
        copy__46: "f32[64]" = torch.ops.aten.copy_.default(primals_339, add_81);  primals_339 = add_81 = None
        copy__47: "i64[]" = torch.ops.aten.copy_.default(primals_340, add_78);  primals_340 = add_78 = None
        copy__48: "f32[96]" = torch.ops.aten.copy_.default(primals_341, add_85);  primals_341 = add_85 = None
        copy__49: "f32[96]" = torch.ops.aten.copy_.default(primals_342, add_86);  primals_342 = add_86 = None
        copy__50: "i64[]" = torch.ops.aten.copy_.default(primals_343, add_83);  primals_343 = add_83 = None
        copy__51: "f32[96]" = torch.ops.aten.copy_.default(primals_344, add_90);  primals_344 = add_90 = None
        copy__52: "f32[96]" = torch.ops.aten.copy_.default(primals_345, add_91);  primals_345 = add_91 = None
        copy__53: "i64[]" = torch.ops.aten.copy_.default(primals_346, add_88);  primals_346 = add_88 = None
        copy__54: "f32[64]" = torch.ops.aten.copy_.default(primals_347, add_95);  primals_347 = add_95 = None
        copy__55: "f32[64]" = torch.ops.aten.copy_.default(primals_348, add_96);  primals_348 = add_96 = None
        copy__56: "i64[]" = torch.ops.aten.copy_.default(primals_349, add_93);  primals_349 = add_93 = None
        copy__57: "f32[64]" = torch.ops.aten.copy_.default(primals_350, add_100);  primals_350 = add_100 = None
        copy__58: "f32[64]" = torch.ops.aten.copy_.default(primals_351, add_101);  primals_351 = add_101 = None
        copy__59: "i64[]" = torch.ops.aten.copy_.default(primals_352, add_98);  primals_352 = add_98 = None
        copy__60: "f32[48]" = torch.ops.aten.copy_.default(primals_353, add_105);  primals_353 = add_105 = None
        copy__61: "f32[48]" = torch.ops.aten.copy_.default(primals_354, add_106);  primals_354 = add_106 = None
        copy__62: "i64[]" = torch.ops.aten.copy_.default(primals_355, add_103);  primals_355 = add_103 = None
        copy__63: "f32[64]" = torch.ops.aten.copy_.default(primals_356, add_110);  primals_356 = add_110 = None
        copy__64: "f32[64]" = torch.ops.aten.copy_.default(primals_357, add_111);  primals_357 = add_111 = None
        copy__65: "i64[]" = torch.ops.aten.copy_.default(primals_358, add_108);  primals_358 = add_108 = None
        copy__66: "f32[64]" = torch.ops.aten.copy_.default(primals_359, add_115);  primals_359 = add_115 = None
        copy__67: "f32[64]" = torch.ops.aten.copy_.default(primals_360, add_116);  primals_360 = add_116 = None
        copy__68: "i64[]" = torch.ops.aten.copy_.default(primals_361, add_113);  primals_361 = add_113 = None
        copy__69: "f32[96]" = torch.ops.aten.copy_.default(primals_362, add_120);  primals_362 = add_120 = None
        copy__70: "f32[96]" = torch.ops.aten.copy_.default(primals_363, add_121);  primals_363 = add_121 = None
        copy__71: "i64[]" = torch.ops.aten.copy_.default(primals_364, add_118);  primals_364 = add_118 = None
        copy__72: "f32[96]" = torch.ops.aten.copy_.default(primals_365, add_125);  primals_365 = add_125 = None
        copy__73: "f32[96]" = torch.ops.aten.copy_.default(primals_366, add_126);  primals_366 = add_126 = None
        copy__74: "i64[]" = torch.ops.aten.copy_.default(primals_367, add_123);  primals_367 = add_123 = None
        copy__75: "f32[64]" = torch.ops.aten.copy_.default(primals_368, add_130);  primals_368 = add_130 = None
        copy__76: "f32[64]" = torch.ops.aten.copy_.default(primals_369, add_131);  primals_369 = add_131 = None
        copy__77: "i64[]" = torch.ops.aten.copy_.default(primals_370, add_128);  primals_370 = add_128 = None
        copy__78: "f32[384]" = torch.ops.aten.copy_.default(primals_371, add_135);  primals_371 = add_135 = None
        copy__79: "f32[384]" = torch.ops.aten.copy_.default(primals_372, add_136);  primals_372 = add_136 = None
        copy__80: "i64[]" = torch.ops.aten.copy_.default(primals_373, add_133);  primals_373 = add_133 = None
        copy__81: "f32[64]" = torch.ops.aten.copy_.default(primals_374, add_140);  primals_374 = add_140 = None
        copy__82: "f32[64]" = torch.ops.aten.copy_.default(primals_375, add_141);  primals_375 = add_141 = None
        copy__83: "i64[]" = torch.ops.aten.copy_.default(primals_376, add_138);  primals_376 = add_138 = None
        copy__84: "f32[96]" = torch.ops.aten.copy_.default(primals_377, add_145);  primals_377 = add_145 = None
        copy__85: "f32[96]" = torch.ops.aten.copy_.default(primals_378, add_146);  primals_378 = add_146 = None
        copy__86: "i64[]" = torch.ops.aten.copy_.default(primals_379, add_143);  primals_379 = add_143 = None
        copy__87: "f32[96]" = torch.ops.aten.copy_.default(primals_380, add_150);  primals_380 = add_150 = None
        copy__88: "f32[96]" = torch.ops.aten.copy_.default(primals_381, add_151);  primals_381 = add_151 = None
        copy__89: "i64[]" = torch.ops.aten.copy_.default(primals_382, add_148);  primals_382 = add_148 = None
        copy__90: "f32[192]" = torch.ops.aten.copy_.default(primals_383, add_155);  primals_383 = add_155 = None
        copy__91: "f32[192]" = torch.ops.aten.copy_.default(primals_384, add_156);  primals_384 = add_156 = None
        copy__92: "i64[]" = torch.ops.aten.copy_.default(primals_385, add_153);  primals_385 = add_153 = None
        copy__93: "f32[128]" = torch.ops.aten.copy_.default(primals_386, add_160);  primals_386 = add_160 = None
        copy__94: "f32[128]" = torch.ops.aten.copy_.default(primals_387, add_161);  primals_387 = add_161 = None
        copy__95: "i64[]" = torch.ops.aten.copy_.default(primals_388, add_158);  primals_388 = add_158 = None
        copy__96: "f32[128]" = torch.ops.aten.copy_.default(primals_389, add_165);  primals_389 = add_165 = None
        copy__97: "f32[128]" = torch.ops.aten.copy_.default(primals_390, add_166);  primals_390 = add_166 = None
        copy__98: "i64[]" = torch.ops.aten.copy_.default(primals_391, add_163);  primals_391 = add_163 = None
        copy__99: "f32[192]" = torch.ops.aten.copy_.default(primals_392, add_170);  primals_392 = add_170 = None
        copy__100: "f32[192]" = torch.ops.aten.copy_.default(primals_393, add_171);  primals_393 = add_171 = None
        copy__101: "i64[]" = torch.ops.aten.copy_.default(primals_394, add_168);  primals_394 = add_168 = None
        copy__102: "f32[128]" = torch.ops.aten.copy_.default(primals_395, add_175);  primals_395 = add_175 = None
        copy__103: "f32[128]" = torch.ops.aten.copy_.default(primals_396, add_176);  primals_396 = add_176 = None
        copy__104: "i64[]" = torch.ops.aten.copy_.default(primals_397, add_173);  primals_397 = add_173 = None
        copy__105: "f32[128]" = torch.ops.aten.copy_.default(primals_398, add_180);  primals_398 = add_180 = None
        copy__106: "f32[128]" = torch.ops.aten.copy_.default(primals_399, add_181);  primals_399 = add_181 = None
        copy__107: "i64[]" = torch.ops.aten.copy_.default(primals_400, add_178);  primals_400 = add_178 = None
        copy__108: "f32[128]" = torch.ops.aten.copy_.default(primals_401, add_185);  primals_401 = add_185 = None
        copy__109: "f32[128]" = torch.ops.aten.copy_.default(primals_402, add_186);  primals_402 = add_186 = None
        copy__110: "i64[]" = torch.ops.aten.copy_.default(primals_403, add_183);  primals_403 = add_183 = None
        copy__111: "f32[128]" = torch.ops.aten.copy_.default(primals_404, add_190);  primals_404 = add_190 = None
        copy__112: "f32[128]" = torch.ops.aten.copy_.default(primals_405, add_191);  primals_405 = add_191 = None
        copy__113: "i64[]" = torch.ops.aten.copy_.default(primals_406, add_188);  primals_406 = add_188 = None
        copy__114: "f32[192]" = torch.ops.aten.copy_.default(primals_407, add_195);  primals_407 = add_195 = None
        copy__115: "f32[192]" = torch.ops.aten.copy_.default(primals_408, add_196);  primals_408 = add_196 = None
        copy__116: "i64[]" = torch.ops.aten.copy_.default(primals_409, add_193);  primals_409 = add_193 = None
        copy__117: "f32[192]" = torch.ops.aten.copy_.default(primals_410, add_200);  primals_410 = add_200 = None
        copy__118: "f32[192]" = torch.ops.aten.copy_.default(primals_411, add_201);  primals_411 = add_201 = None
        copy__119: "i64[]" = torch.ops.aten.copy_.default(primals_412, add_198);  primals_412 = add_198 = None
        copy__120: "f32[192]" = torch.ops.aten.copy_.default(primals_413, add_205);  primals_413 = add_205 = None
        copy__121: "f32[192]" = torch.ops.aten.copy_.default(primals_414, add_206);  primals_414 = add_206 = None
        copy__122: "i64[]" = torch.ops.aten.copy_.default(primals_415, add_203);  primals_415 = add_203 = None
        copy__123: "f32[160]" = torch.ops.aten.copy_.default(primals_416, add_210);  primals_416 = add_210 = None
        copy__124: "f32[160]" = torch.ops.aten.copy_.default(primals_417, add_211);  primals_417 = add_211 = None
        copy__125: "i64[]" = torch.ops.aten.copy_.default(primals_418, add_208);  primals_418 = add_208 = None
        copy__126: "f32[160]" = torch.ops.aten.copy_.default(primals_419, add_215);  primals_419 = add_215 = None
        copy__127: "f32[160]" = torch.ops.aten.copy_.default(primals_420, add_216);  primals_420 = add_216 = None
        copy__128: "i64[]" = torch.ops.aten.copy_.default(primals_421, add_213);  primals_421 = add_213 = None
        copy__129: "f32[192]" = torch.ops.aten.copy_.default(primals_422, add_220);  primals_422 = add_220 = None
        copy__130: "f32[192]" = torch.ops.aten.copy_.default(primals_423, add_221);  primals_423 = add_221 = None
        copy__131: "i64[]" = torch.ops.aten.copy_.default(primals_424, add_218);  primals_424 = add_218 = None
        copy__132: "f32[160]" = torch.ops.aten.copy_.default(primals_425, add_225);  primals_425 = add_225 = None
        copy__133: "f32[160]" = torch.ops.aten.copy_.default(primals_426, add_226);  primals_426 = add_226 = None
        copy__134: "i64[]" = torch.ops.aten.copy_.default(primals_427, add_223);  primals_427 = add_223 = None
        copy__135: "f32[160]" = torch.ops.aten.copy_.default(primals_428, add_230);  primals_428 = add_230 = None
        copy__136: "f32[160]" = torch.ops.aten.copy_.default(primals_429, add_231);  primals_429 = add_231 = None
        copy__137: "i64[]" = torch.ops.aten.copy_.default(primals_430, add_228);  primals_430 = add_228 = None
        copy__138: "f32[160]" = torch.ops.aten.copy_.default(primals_431, add_235);  primals_431 = add_235 = None
        copy__139: "f32[160]" = torch.ops.aten.copy_.default(primals_432, add_236);  primals_432 = add_236 = None
        copy__140: "i64[]" = torch.ops.aten.copy_.default(primals_433, add_233);  primals_433 = add_233 = None
        copy__141: "f32[160]" = torch.ops.aten.copy_.default(primals_434, add_240);  primals_434 = add_240 = None
        copy__142: "f32[160]" = torch.ops.aten.copy_.default(primals_435, add_241);  primals_435 = add_241 = None
        copy__143: "i64[]" = torch.ops.aten.copy_.default(primals_436, add_238);  primals_436 = add_238 = None
        copy__144: "f32[192]" = torch.ops.aten.copy_.default(primals_437, add_245);  primals_437 = add_245 = None
        copy__145: "f32[192]" = torch.ops.aten.copy_.default(primals_438, add_246);  primals_438 = add_246 = None
        copy__146: "i64[]" = torch.ops.aten.copy_.default(primals_439, add_243);  primals_439 = add_243 = None
        copy__147: "f32[192]" = torch.ops.aten.copy_.default(primals_440, add_250);  primals_440 = add_250 = None
        copy__148: "f32[192]" = torch.ops.aten.copy_.default(primals_441, add_251);  primals_441 = add_251 = None
        copy__149: "i64[]" = torch.ops.aten.copy_.default(primals_442, add_248);  primals_442 = add_248 = None
        copy__150: "f32[192]" = torch.ops.aten.copy_.default(primals_443, add_255);  primals_443 = add_255 = None
        copy__151: "f32[192]" = torch.ops.aten.copy_.default(primals_444, add_256);  primals_444 = add_256 = None
        copy__152: "i64[]" = torch.ops.aten.copy_.default(primals_445, add_253);  primals_445 = add_253 = None
        copy__153: "f32[160]" = torch.ops.aten.copy_.default(primals_446, add_260);  primals_446 = add_260 = None
        copy__154: "f32[160]" = torch.ops.aten.copy_.default(primals_447, add_261);  primals_447 = add_261 = None
        copy__155: "i64[]" = torch.ops.aten.copy_.default(primals_448, add_258);  primals_448 = add_258 = None
        copy__156: "f32[160]" = torch.ops.aten.copy_.default(primals_449, add_265);  primals_449 = add_265 = None
        copy__157: "f32[160]" = torch.ops.aten.copy_.default(primals_450, add_266);  primals_450 = add_266 = None
        copy__158: "i64[]" = torch.ops.aten.copy_.default(primals_451, add_263);  primals_451 = add_263 = None
        copy__159: "f32[192]" = torch.ops.aten.copy_.default(primals_452, add_270);  primals_452 = add_270 = None
        copy__160: "f32[192]" = torch.ops.aten.copy_.default(primals_453, add_271);  primals_453 = add_271 = None
        copy__161: "i64[]" = torch.ops.aten.copy_.default(primals_454, add_268);  primals_454 = add_268 = None
        copy__162: "f32[160]" = torch.ops.aten.copy_.default(primals_455, add_275);  primals_455 = add_275 = None
        copy__163: "f32[160]" = torch.ops.aten.copy_.default(primals_456, add_276);  primals_456 = add_276 = None
        copy__164: "i64[]" = torch.ops.aten.copy_.default(primals_457, add_273);  primals_457 = add_273 = None
        copy__165: "f32[160]" = torch.ops.aten.copy_.default(primals_458, add_280);  primals_458 = add_280 = None
        copy__166: "f32[160]" = torch.ops.aten.copy_.default(primals_459, add_281);  primals_459 = add_281 = None
        copy__167: "i64[]" = torch.ops.aten.copy_.default(primals_460, add_278);  primals_460 = add_278 = None
        copy__168: "f32[160]" = torch.ops.aten.copy_.default(primals_461, add_285);  primals_461 = add_285 = None
        copy__169: "f32[160]" = torch.ops.aten.copy_.default(primals_462, add_286);  primals_462 = add_286 = None
        copy__170: "i64[]" = torch.ops.aten.copy_.default(primals_463, add_283);  primals_463 = add_283 = None
        copy__171: "f32[160]" = torch.ops.aten.copy_.default(primals_464, add_290);  primals_464 = add_290 = None
        copy__172: "f32[160]" = torch.ops.aten.copy_.default(primals_465, add_291);  primals_465 = add_291 = None
        copy__173: "i64[]" = torch.ops.aten.copy_.default(primals_466, add_288);  primals_466 = add_288 = None
        copy__174: "f32[192]" = torch.ops.aten.copy_.default(primals_467, add_295);  primals_467 = add_295 = None
        copy__175: "f32[192]" = torch.ops.aten.copy_.default(primals_468, add_296);  primals_468 = add_296 = None
        copy__176: "i64[]" = torch.ops.aten.copy_.default(primals_469, add_293);  primals_469 = add_293 = None
        copy__177: "f32[192]" = torch.ops.aten.copy_.default(primals_470, add_300);  primals_470 = add_300 = None
        copy__178: "f32[192]" = torch.ops.aten.copy_.default(primals_471, add_301);  primals_471 = add_301 = None
        copy__179: "i64[]" = torch.ops.aten.copy_.default(primals_472, add_298);  primals_472 = add_298 = None
        copy__180: "f32[192]" = torch.ops.aten.copy_.default(primals_473, add_305);  primals_473 = add_305 = None
        copy__181: "f32[192]" = torch.ops.aten.copy_.default(primals_474, add_306);  primals_474 = add_306 = None
        copy__182: "i64[]" = torch.ops.aten.copy_.default(primals_475, add_303);  primals_475 = add_303 = None
        copy__183: "f32[192]" = torch.ops.aten.copy_.default(primals_476, add_310);  primals_476 = add_310 = None
        copy__184: "f32[192]" = torch.ops.aten.copy_.default(primals_477, add_311);  primals_477 = add_311 = None
        copy__185: "i64[]" = torch.ops.aten.copy_.default(primals_478, add_308);  primals_478 = add_308 = None
        copy__186: "f32[192]" = torch.ops.aten.copy_.default(primals_479, add_315);  primals_479 = add_315 = None
        copy__187: "f32[192]" = torch.ops.aten.copy_.default(primals_480, add_316);  primals_480 = add_316 = None
        copy__188: "i64[]" = torch.ops.aten.copy_.default(primals_481, add_313);  primals_481 = add_313 = None
        copy__189: "f32[192]" = torch.ops.aten.copy_.default(primals_482, add_320);  primals_482 = add_320 = None
        copy__190: "f32[192]" = torch.ops.aten.copy_.default(primals_483, add_321);  primals_483 = add_321 = None
        copy__191: "i64[]" = torch.ops.aten.copy_.default(primals_484, add_318);  primals_484 = add_318 = None
        copy__192: "f32[192]" = torch.ops.aten.copy_.default(primals_485, add_325);  primals_485 = add_325 = None
        copy__193: "f32[192]" = torch.ops.aten.copy_.default(primals_486, add_326);  primals_486 = add_326 = None
        copy__194: "i64[]" = torch.ops.aten.copy_.default(primals_487, add_323);  primals_487 = add_323 = None
        copy__195: "f32[192]" = torch.ops.aten.copy_.default(primals_488, add_330);  primals_488 = add_330 = None
        copy__196: "f32[192]" = torch.ops.aten.copy_.default(primals_489, add_331);  primals_489 = add_331 = None
        copy__197: "i64[]" = torch.ops.aten.copy_.default(primals_490, add_328);  primals_490 = add_328 = None
        copy__198: "f32[192]" = torch.ops.aten.copy_.default(primals_491, add_335);  primals_491 = add_335 = None
        copy__199: "f32[192]" = torch.ops.aten.copy_.default(primals_492, add_336);  primals_492 = add_336 = None
        copy__200: "i64[]" = torch.ops.aten.copy_.default(primals_493, add_333);  primals_493 = add_333 = None
        copy__201: "f32[192]" = torch.ops.aten.copy_.default(primals_494, add_340);  primals_494 = add_340 = None
        copy__202: "f32[192]" = torch.ops.aten.copy_.default(primals_495, add_341);  primals_495 = add_341 = None
        copy__203: "i64[]" = torch.ops.aten.copy_.default(primals_496, add_338);  primals_496 = add_338 = None
        copy__204: "f32[192]" = torch.ops.aten.copy_.default(primals_497, add_345);  primals_497 = add_345 = None
        copy__205: "f32[192]" = torch.ops.aten.copy_.default(primals_498, add_346);  primals_498 = add_346 = None
        copy__206: "i64[]" = torch.ops.aten.copy_.default(primals_499, add_343);  primals_499 = add_343 = None
        copy__207: "f32[192]" = torch.ops.aten.copy_.default(primals_500, add_350);  primals_500 = add_350 = None
        copy__208: "f32[192]" = torch.ops.aten.copy_.default(primals_501, add_351);  primals_501 = add_351 = None
        copy__209: "i64[]" = torch.ops.aten.copy_.default(primals_502, add_348);  primals_502 = add_348 = None
        copy__210: "f32[128]" = torch.ops.aten.copy_.default(primals_503, add_355);  primals_503 = add_355 = None
        copy__211: "f32[128]" = torch.ops.aten.copy_.default(primals_504, add_356);  primals_504 = add_356 = None
        copy__212: "i64[]" = torch.ops.aten.copy_.default(primals_505, add_353);  primals_505 = add_353 = None
        copy__213: "f32[768]" = torch.ops.aten.copy_.default(primals_506, add_360);  primals_506 = add_360 = None
        copy__214: "f32[768]" = torch.ops.aten.copy_.default(primals_507, add_361);  primals_507 = add_361 = None
        copy__215: "i64[]" = torch.ops.aten.copy_.default(primals_508, add_358);  primals_508 = add_358 = None
        copy__216: "f32[192]" = torch.ops.aten.copy_.default(primals_509, add_365);  primals_509 = add_365 = None
        copy__217: "f32[192]" = torch.ops.aten.copy_.default(primals_510, add_366);  primals_510 = add_366 = None
        copy__218: "i64[]" = torch.ops.aten.copy_.default(primals_511, add_363);  primals_511 = add_363 = None
        copy__219: "f32[320]" = torch.ops.aten.copy_.default(primals_512, add_370);  primals_512 = add_370 = None
        copy__220: "f32[320]" = torch.ops.aten.copy_.default(primals_513, add_371);  primals_513 = add_371 = None
        copy__221: "i64[]" = torch.ops.aten.copy_.default(primals_514, add_368);  primals_514 = add_368 = None
        copy__222: "f32[192]" = torch.ops.aten.copy_.default(primals_515, add_375);  primals_515 = add_375 = None
        copy__223: "f32[192]" = torch.ops.aten.copy_.default(primals_516, add_376);  primals_516 = add_376 = None
        copy__224: "i64[]" = torch.ops.aten.copy_.default(primals_517, add_373);  primals_517 = add_373 = None
        copy__225: "f32[192]" = torch.ops.aten.copy_.default(primals_518, add_380);  primals_518 = add_380 = None
        copy__226: "f32[192]" = torch.ops.aten.copy_.default(primals_519, add_381);  primals_519 = add_381 = None
        copy__227: "i64[]" = torch.ops.aten.copy_.default(primals_520, add_378);  primals_520 = add_378 = None
        copy__228: "f32[192]" = torch.ops.aten.copy_.default(primals_521, add_385);  primals_521 = add_385 = None
        copy__229: "f32[192]" = torch.ops.aten.copy_.default(primals_522, add_386);  primals_522 = add_386 = None
        copy__230: "i64[]" = torch.ops.aten.copy_.default(primals_523, add_383);  primals_523 = add_383 = None
        copy__231: "f32[192]" = torch.ops.aten.copy_.default(primals_524, add_390);  primals_524 = add_390 = None
        copy__232: "f32[192]" = torch.ops.aten.copy_.default(primals_525, add_391);  primals_525 = add_391 = None
        copy__233: "i64[]" = torch.ops.aten.copy_.default(primals_526, add_388);  primals_526 = add_388 = None
        copy__234: "f32[320]" = torch.ops.aten.copy_.default(primals_527, add_395);  primals_527 = add_395 = None
        copy__235: "f32[320]" = torch.ops.aten.copy_.default(primals_528, add_396);  primals_528 = add_396 = None
        copy__236: "i64[]" = torch.ops.aten.copy_.default(primals_529, add_393);  primals_529 = add_393 = None
        copy__237: "f32[384]" = torch.ops.aten.copy_.default(primals_530, add_400);  primals_530 = add_400 = None
        copy__238: "f32[384]" = torch.ops.aten.copy_.default(primals_531, add_401);  primals_531 = add_401 = None
        copy__239: "i64[]" = torch.ops.aten.copy_.default(primals_532, add_398);  primals_532 = add_398 = None
        copy__240: "f32[384]" = torch.ops.aten.copy_.default(primals_533, add_405);  primals_533 = add_405 = None
        copy__241: "f32[384]" = torch.ops.aten.copy_.default(primals_534, add_406);  primals_534 = add_406 = None
        copy__242: "i64[]" = torch.ops.aten.copy_.default(primals_535, add_403);  primals_535 = add_403 = None
        copy__243: "f32[384]" = torch.ops.aten.copy_.default(primals_536, add_410);  primals_536 = add_410 = None
        copy__244: "f32[384]" = torch.ops.aten.copy_.default(primals_537, add_411);  primals_537 = add_411 = None
        copy__245: "i64[]" = torch.ops.aten.copy_.default(primals_538, add_408);  primals_538 = add_408 = None
        copy__246: "f32[448]" = torch.ops.aten.copy_.default(primals_539, add_415);  primals_539 = add_415 = None
        copy__247: "f32[448]" = torch.ops.aten.copy_.default(primals_540, add_416);  primals_540 = add_416 = None
        copy__248: "i64[]" = torch.ops.aten.copy_.default(primals_541, add_413);  primals_541 = add_413 = None
        copy__249: "f32[384]" = torch.ops.aten.copy_.default(primals_542, add_420);  primals_542 = add_420 = None
        copy__250: "f32[384]" = torch.ops.aten.copy_.default(primals_543, add_421);  primals_543 = add_421 = None
        copy__251: "i64[]" = torch.ops.aten.copy_.default(primals_544, add_418);  primals_544 = add_418 = None
        copy__252: "f32[384]" = torch.ops.aten.copy_.default(primals_545, add_425);  primals_545 = add_425 = None
        copy__253: "f32[384]" = torch.ops.aten.copy_.default(primals_546, add_426);  primals_546 = add_426 = None
        copy__254: "i64[]" = torch.ops.aten.copy_.default(primals_547, add_423);  primals_547 = add_423 = None
        copy__255: "f32[384]" = torch.ops.aten.copy_.default(primals_548, add_430);  primals_548 = add_430 = None
        copy__256: "f32[384]" = torch.ops.aten.copy_.default(primals_549, add_431);  primals_549 = add_431 = None
        copy__257: "i64[]" = torch.ops.aten.copy_.default(primals_550, add_428);  primals_550 = add_428 = None
        copy__258: "f32[192]" = torch.ops.aten.copy_.default(primals_551, add_435);  primals_551 = add_435 = None
        copy__259: "f32[192]" = torch.ops.aten.copy_.default(primals_552, add_436);  primals_552 = add_436 = None
        copy__260: "i64[]" = torch.ops.aten.copy_.default(primals_553, add_433);  primals_553 = add_433 = None
        copy__261: "f32[320]" = torch.ops.aten.copy_.default(primals_554, add_440);  primals_554 = add_440 = None
        copy__262: "f32[320]" = torch.ops.aten.copy_.default(primals_555, add_441);  primals_555 = add_441 = None
        copy__263: "i64[]" = torch.ops.aten.copy_.default(primals_556, add_438);  primals_556 = add_438 = None
        copy__264: "f32[384]" = torch.ops.aten.copy_.default(primals_557, add_445);  primals_557 = add_445 = None
        copy__265: "f32[384]" = torch.ops.aten.copy_.default(primals_558, add_446);  primals_558 = add_446 = None
        copy__266: "i64[]" = torch.ops.aten.copy_.default(primals_559, add_443);  primals_559 = add_443 = None
        copy__267: "f32[384]" = torch.ops.aten.copy_.default(primals_560, add_450);  primals_560 = add_450 = None
        copy__268: "f32[384]" = torch.ops.aten.copy_.default(primals_561, add_451);  primals_561 = add_451 = None
        copy__269: "i64[]" = torch.ops.aten.copy_.default(primals_562, add_448);  primals_562 = add_448 = None
        copy__270: "f32[384]" = torch.ops.aten.copy_.default(primals_563, add_455);  primals_563 = add_455 = None
        copy__271: "f32[384]" = torch.ops.aten.copy_.default(primals_564, add_456);  primals_564 = add_456 = None
        copy__272: "i64[]" = torch.ops.aten.copy_.default(primals_565, add_453);  primals_565 = add_453 = None
        copy__273: "f32[448]" = torch.ops.aten.copy_.default(primals_566, add_460);  primals_566 = add_460 = None
        copy__274: "f32[448]" = torch.ops.aten.copy_.default(primals_567, add_461);  primals_567 = add_461 = None
        copy__275: "i64[]" = torch.ops.aten.copy_.default(primals_568, add_458);  primals_568 = add_458 = None
        copy__276: "f32[384]" = torch.ops.aten.copy_.default(primals_569, add_465);  primals_569 = add_465 = None
        copy__277: "f32[384]" = torch.ops.aten.copy_.default(primals_570, add_466);  primals_570 = add_466 = None
        copy__278: "i64[]" = torch.ops.aten.copy_.default(primals_571, add_463);  primals_571 = add_463 = None
        copy__279: "f32[384]" = torch.ops.aten.copy_.default(primals_572, add_470);  primals_572 = add_470 = None
        copy__280: "f32[384]" = torch.ops.aten.copy_.default(primals_573, add_471);  primals_573 = add_471 = None
        copy__281: "i64[]" = torch.ops.aten.copy_.default(primals_574, add_468);  primals_574 = add_468 = None
        copy__282: "f32[384]" = torch.ops.aten.copy_.default(primals_575, add_475);  primals_575 = add_475 = None
        copy__283: "f32[384]" = torch.ops.aten.copy_.default(primals_576, add_476);  primals_576 = add_476 = None
        copy__284: "i64[]" = torch.ops.aten.copy_.default(primals_577, add_473);  primals_577 = add_473 = None
        copy__285: "f32[192]" = torch.ops.aten.copy_.default(primals_578, add_480);  primals_578 = add_480 = None
        copy__286: "f32[192]" = torch.ops.aten.copy_.default(primals_579, add_481);  primals_579 = add_481 = None
        copy__287: "i64[]" = torch.ops.aten.copy_.default(primals_580, add_478);  primals_580 = add_478 = None
        return [addmm_1, addmm, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, cat, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, getitem_6, getitem_7, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, getitem_12, getitem_13, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_6, convolution_7, squeeze_22, convolution_8, squeeze_25, relu_8, convolution_9, squeeze_28, relu_9, convolution_10, squeeze_31, avg_pool2d, convolution_11, squeeze_34, cat_1, convolution_12, squeeze_37, convolution_13, squeeze_40, relu_13, convolution_14, squeeze_43, convolution_15, squeeze_46, relu_15, convolution_16, squeeze_49, relu_16, convolution_17, squeeze_52, avg_pool2d_1, convolution_18, squeeze_55, cat_2, convolution_19, squeeze_58, convolution_20, squeeze_61, relu_20, convolution_21, squeeze_64, convolution_22, squeeze_67, relu_22, convolution_23, squeeze_70, relu_23, convolution_24, squeeze_73, avg_pool2d_2, convolution_25, squeeze_76, cat_3, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_27, convolution_28, squeeze_85, relu_28, convolution_29, squeeze_88, getitem_65, cat_4, convolution_30, squeeze_91, convolution_31, squeeze_94, relu_31, convolution_32, squeeze_97, relu_32, convolution_33, squeeze_100, convolution_34, squeeze_103, relu_34, convolution_35, squeeze_106, relu_35, convolution_36, squeeze_109, relu_36, convolution_37, squeeze_112, relu_37, convolution_38, squeeze_115, avg_pool2d_3, convolution_39, squeeze_118, cat_5, convolution_40, squeeze_121, convolution_41, squeeze_124, relu_41, convolution_42, squeeze_127, relu_42, convolution_43, squeeze_130, convolution_44, squeeze_133, relu_44, convolution_45, squeeze_136, relu_45, convolution_46, squeeze_139, relu_46, convolution_47, squeeze_142, relu_47, convolution_48, squeeze_145, avg_pool2d_4, convolution_49, squeeze_148, cat_6, convolution_50, squeeze_151, convolution_51, squeeze_154, relu_51, convolution_52, squeeze_157, relu_52, convolution_53, squeeze_160, convolution_54, squeeze_163, relu_54, convolution_55, squeeze_166, relu_55, convolution_56, squeeze_169, relu_56, convolution_57, squeeze_172, relu_57, convolution_58, squeeze_175, avg_pool2d_5, convolution_59, squeeze_178, cat_7, convolution_60, squeeze_181, convolution_61, squeeze_184, relu_61, convolution_62, squeeze_187, relu_62, convolution_63, squeeze_190, convolution_64, squeeze_193, relu_64, convolution_65, squeeze_196, relu_65, convolution_66, squeeze_199, relu_66, convolution_67, squeeze_202, relu_67, convolution_68, squeeze_205, avg_pool2d_6, convolution_69, squeeze_208, cat_8, avg_pool2d_7, convolution_70, squeeze_211, relu_70, convolution_71, squeeze_214, view, convolution_72, squeeze_217, relu_72, convolution_73, squeeze_220, convolution_74, squeeze_223, relu_74, convolution_75, squeeze_226, relu_75, convolution_76, squeeze_229, relu_76, convolution_77, squeeze_232, getitem_163, cat_9, convolution_78, squeeze_235, convolution_79, squeeze_238, relu_79, convolution_80, squeeze_241, convolution_81, squeeze_244, convolution_82, squeeze_247, relu_82, convolution_83, squeeze_250, relu_83, convolution_84, squeeze_253, convolution_85, squeeze_256, avg_pool2d_8, convolution_86, squeeze_259, cat_12, convolution_87, squeeze_262, convolution_88, squeeze_265, relu_88, convolution_89, squeeze_268, convolution_90, squeeze_271, convolution_91, squeeze_274, relu_91, convolution_92, squeeze_277, relu_92, convolution_93, squeeze_280, convolution_94, squeeze_283, avg_pool2d_9, convolution_95, squeeze_286, gt, view_1, permute_2, le, unsqueeze_389, le_1, unsqueeze_401, le_2, unsqueeze_413, unsqueeze_425, unsqueeze_437, le_5, unsqueeze_449, le_6, unsqueeze_461, unsqueeze_473, le_8, unsqueeze_485, le_9, unsqueeze_497, le_10, unsqueeze_509, le_11, unsqueeze_521, unsqueeze_533, unsqueeze_545, le_14, unsqueeze_557, le_15, unsqueeze_569, unsqueeze_581, le_17, unsqueeze_593, le_18, unsqueeze_605, unsqueeze_617, unsqueeze_629, unsqueeze_641, le_22, unsqueeze_653, unsqueeze_665, permute_6, le_24, unsqueeze_677, unsqueeze_689, le_26, unsqueeze_701, le_27, unsqueeze_713, unsqueeze_725, unsqueeze_737, unsqueeze_749, unsqueeze_761, le_32, unsqueeze_773, unsqueeze_785, unsqueeze_797, le_35, unsqueeze_809, le_36, unsqueeze_821, le_37, unsqueeze_833, unsqueeze_845, unsqueeze_857, unsqueeze_869, unsqueeze_881, le_42, unsqueeze_893, unsqueeze_905, unsqueeze_917, le_45, unsqueeze_929, le_46, unsqueeze_941, le_47, unsqueeze_953, unsqueeze_965, unsqueeze_977, unsqueeze_989, unsqueeze_1001, le_52, unsqueeze_1013, unsqueeze_1025, unsqueeze_1037, le_55, unsqueeze_1049, le_56, unsqueeze_1061, le_57, unsqueeze_1073, unsqueeze_1085, unsqueeze_1097, unsqueeze_1109, unsqueeze_1121, le_62, unsqueeze_1133, unsqueeze_1145, unsqueeze_1157, le_65, unsqueeze_1169, le_66, unsqueeze_1181, unsqueeze_1193, unsqueeze_1205, le_69, unsqueeze_1217, le_70, unsqueeze_1229, le_71, unsqueeze_1241, unsqueeze_1253, unsqueeze_1265, le_74, unsqueeze_1277, unsqueeze_1289, le_76, unsqueeze_1301, le_77, unsqueeze_1313, le_78, unsqueeze_1325, unsqueeze_1337, unsqueeze_1349, le_81, unsqueeze_1361, unsqueeze_1373, le_83, unsqueeze_1385, le_84, unsqueeze_1397, le_85, unsqueeze_1409, unsqueeze_1421, unsqueeze_1433, le_88, unsqueeze_1445, unsqueeze_1457, le_90, unsqueeze_1469, unsqueeze_1481, unsqueeze_1493, unsqueeze_1505, unsqueeze_1517, unsqueeze_1529]
        