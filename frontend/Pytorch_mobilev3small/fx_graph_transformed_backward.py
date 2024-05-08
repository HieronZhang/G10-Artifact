class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[16, 3, 3, 3]", primals_2: "f32[16]", primals_4: "f32[16, 1, 3, 3]", primals_5: "f32[16]", primals_7: "f32[8, 16, 1, 1]", primals_9: "f32[16, 8, 1, 1]", primals_11: "f32[16, 16, 1, 1]", primals_12: "f32[16]", primals_14: "f32[72, 16, 1, 1]", primals_15: "f32[72]", primals_17: "f32[72, 1, 3, 3]", primals_18: "f32[72]", primals_20: "f32[24, 72, 1, 1]", primals_21: "f32[24]", primals_23: "f32[88, 24, 1, 1]", primals_24: "f32[88]", primals_26: "f32[88, 1, 3, 3]", primals_27: "f32[88]", primals_29: "f32[24, 88, 1, 1]", primals_30: "f32[24]", primals_32: "f32[96, 24, 1, 1]", primals_33: "f32[96]", primals_35: "f32[96, 1, 5, 5]", primals_36: "f32[96]", primals_38: "f32[24, 96, 1, 1]", primals_40: "f32[96, 24, 1, 1]", primals_42: "f32[40, 96, 1, 1]", primals_43: "f32[40]", primals_45: "f32[240, 40, 1, 1]", primals_46: "f32[240]", primals_48: "f32[240, 1, 5, 5]", primals_49: "f32[240]", primals_51: "f32[64, 240, 1, 1]", primals_53: "f32[240, 64, 1, 1]", primals_55: "f32[40, 240, 1, 1]", primals_56: "f32[40]", primals_58: "f32[240, 40, 1, 1]", primals_59: "f32[240]", primals_61: "f32[240, 1, 5, 5]", primals_62: "f32[240]", primals_64: "f32[64, 240, 1, 1]", primals_66: "f32[240, 64, 1, 1]", primals_68: "f32[40, 240, 1, 1]", primals_69: "f32[40]", primals_71: "f32[120, 40, 1, 1]", primals_72: "f32[120]", primals_74: "f32[120, 1, 5, 5]", primals_75: "f32[120]", primals_77: "f32[32, 120, 1, 1]", primals_79: "f32[120, 32, 1, 1]", primals_81: "f32[48, 120, 1, 1]", primals_82: "f32[48]", primals_84: "f32[144, 48, 1, 1]", primals_85: "f32[144]", primals_87: "f32[144, 1, 5, 5]", primals_88: "f32[144]", primals_90: "f32[40, 144, 1, 1]", primals_92: "f32[144, 40, 1, 1]", primals_94: "f32[48, 144, 1, 1]", primals_95: "f32[48]", primals_97: "f32[288, 48, 1, 1]", primals_98: "f32[288]", primals_100: "f32[288, 1, 5, 5]", primals_101: "f32[288]", primals_103: "f32[72, 288, 1, 1]", primals_105: "f32[288, 72, 1, 1]", primals_107: "f32[96, 288, 1, 1]", primals_108: "f32[96]", primals_110: "f32[576, 96, 1, 1]", primals_111: "f32[576]", primals_113: "f32[576, 1, 5, 5]", primals_114: "f32[576]", primals_116: "f32[144, 576, 1, 1]", primals_118: "f32[576, 144, 1, 1]", primals_120: "f32[96, 576, 1, 1]", primals_121: "f32[96]", primals_123: "f32[576, 96, 1, 1]", primals_124: "f32[576]", primals_126: "f32[576, 1, 5, 5]", primals_127: "f32[576]", primals_129: "f32[144, 576, 1, 1]", primals_131: "f32[576, 144, 1, 1]", primals_133: "f32[96, 576, 1, 1]", primals_134: "f32[96]", primals_136: "f32[576, 96, 1, 1]", primals_137: "f32[576]", primals_245: "f32[64, 3, 224, 224]", convolution: "f32[64, 16, 112, 112]", squeeze_1: "f32[16]", clone_102: "f32[64, 16, 112, 112]", div: "f32[64, 16, 112, 112]", convolution_1: "f32[64, 16, 56, 56]", squeeze_4: "f32[16]", relu: "f32[64, 16, 56, 56]", mean: "f32[64, 16, 1, 1]", relu_1: "f32[64, 8, 1, 1]", div_1: "f32[64, 16, 1, 1]", mul_15: "f32[64, 16, 56, 56]", convolution_4: "f32[64, 16, 56, 56]", squeeze_7: "f32[16]", add_16: "f32[64, 16, 56, 56]", convolution_5: "f32[64, 72, 56, 56]", squeeze_10: "f32[72]", relu_2: "f32[64, 72, 56, 56]", convolution_6: "f32[64, 72, 28, 28]", squeeze_13: "f32[72]", relu_3: "f32[64, 72, 28, 28]", convolution_7: "f32[64, 24, 28, 28]", squeeze_16: "f32[24]", add_31: "f32[64, 24, 28, 28]", convolution_8: "f32[64, 88, 28, 28]", squeeze_19: "f32[88]", relu_4: "f32[64, 88, 28, 28]", convolution_9: "f32[64, 88, 28, 28]", squeeze_22: "f32[88]", relu_5: "f32[64, 88, 28, 28]", convolution_10: "f32[64, 24, 28, 28]", squeeze_25: "f32[24]", add_47: "f32[64, 24, 28, 28]", convolution_11: "f32[64, 96, 28, 28]", squeeze_28: "f32[96]", clone_103: "f32[64, 96, 28, 28]", div_2: "f32[64, 96, 28, 28]", convolution_12: "f32[64, 96, 14, 14]", squeeze_31: "f32[96]", clone_104: "f32[64, 96, 14, 14]", div_3: "f32[64, 96, 14, 14]", mean_1: "f32[64, 96, 1, 1]", relu_6: "f32[64, 24, 1, 1]", div_4: "f32[64, 96, 1, 1]", mul_81: "f32[64, 96, 14, 14]", convolution_15: "f32[64, 40, 14, 14]", squeeze_34: "f32[40]", add_65: "f32[64, 40, 14, 14]", convolution_16: "f32[64, 240, 14, 14]", squeeze_37: "f32[240]", clone_105: "f32[64, 240, 14, 14]", div_5: "f32[64, 240, 14, 14]", convolution_17: "f32[64, 240, 14, 14]", squeeze_40: "f32[240]", clone_106: "f32[64, 240, 14, 14]", div_6: "f32[64, 240, 14, 14]", mean_2: "f32[64, 240, 1, 1]", relu_7: "f32[64, 64, 1, 1]", div_7: "f32[64, 240, 1, 1]", mul_105: "f32[64, 240, 14, 14]", convolution_20: "f32[64, 40, 14, 14]", squeeze_43: "f32[40]", add_84: "f32[64, 40, 14, 14]", convolution_21: "f32[64, 240, 14, 14]", squeeze_46: "f32[240]", clone_107: "f32[64, 240, 14, 14]", div_8: "f32[64, 240, 14, 14]", convolution_22: "f32[64, 240, 14, 14]", squeeze_49: "f32[240]", clone_108: "f32[64, 240, 14, 14]", div_9: "f32[64, 240, 14, 14]", mean_3: "f32[64, 240, 1, 1]", relu_8: "f32[64, 64, 1, 1]", div_10: "f32[64, 240, 1, 1]", mul_129: "f32[64, 240, 14, 14]", convolution_25: "f32[64, 40, 14, 14]", squeeze_52: "f32[40]", add_103: "f32[64, 40, 14, 14]", convolution_26: "f32[64, 120, 14, 14]", squeeze_55: "f32[120]", clone_109: "f32[64, 120, 14, 14]", div_11: "f32[64, 120, 14, 14]", convolution_27: "f32[64, 120, 14, 14]", squeeze_58: "f32[120]", clone_110: "f32[64, 120, 14, 14]", div_12: "f32[64, 120, 14, 14]", mean_4: "f32[64, 120, 1, 1]", relu_9: "f32[64, 32, 1, 1]", div_13: "f32[64, 120, 1, 1]", mul_153: "f32[64, 120, 14, 14]", convolution_30: "f32[64, 48, 14, 14]", squeeze_61: "f32[48]", add_121: "f32[64, 48, 14, 14]", convolution_31: "f32[64, 144, 14, 14]", squeeze_64: "f32[144]", clone_111: "f32[64, 144, 14, 14]", div_14: "f32[64, 144, 14, 14]", convolution_32: "f32[64, 144, 14, 14]", squeeze_67: "f32[144]", clone_112: "f32[64, 144, 14, 14]", div_15: "f32[64, 144, 14, 14]", mean_5: "f32[64, 144, 1, 1]", relu_10: "f32[64, 40, 1, 1]", div_16: "f32[64, 144, 1, 1]", mul_177: "f32[64, 144, 14, 14]", convolution_35: "f32[64, 48, 14, 14]", squeeze_70: "f32[48]", add_140: "f32[64, 48, 14, 14]", convolution_36: "f32[64, 288, 14, 14]", squeeze_73: "f32[288]", clone_113: "f32[64, 288, 14, 14]", div_17: "f32[64, 288, 14, 14]", convolution_37: "f32[64, 288, 7, 7]", squeeze_76: "f32[288]", clone_114: "f32[64, 288, 7, 7]", div_18: "f32[64, 288, 7, 7]", mean_6: "f32[64, 288, 1, 1]", relu_11: "f32[64, 72, 1, 1]", div_19: "f32[64, 288, 1, 1]", mul_201: "f32[64, 288, 7, 7]", convolution_40: "f32[64, 96, 7, 7]", squeeze_79: "f32[96]", add_158: "f32[64, 96, 7, 7]", convolution_41: "f32[64, 576, 7, 7]", squeeze_82: "f32[576]", clone_115: "f32[64, 576, 7, 7]", div_20: "f32[64, 576, 7, 7]", convolution_42: "f32[64, 576, 7, 7]", squeeze_85: "f32[576]", clone_116: "f32[64, 576, 7, 7]", div_21: "f32[64, 576, 7, 7]", mean_7: "f32[64, 576, 1, 1]", relu_12: "f32[64, 144, 1, 1]", div_22: "f32[64, 576, 1, 1]", mul_225: "f32[64, 576, 7, 7]", convolution_45: "f32[64, 96, 7, 7]", squeeze_88: "f32[96]", add_177: "f32[64, 96, 7, 7]", convolution_46: "f32[64, 576, 7, 7]", squeeze_91: "f32[576]", clone_117: "f32[64, 576, 7, 7]", div_23: "f32[64, 576, 7, 7]", convolution_47: "f32[64, 576, 7, 7]", squeeze_94: "f32[576]", clone_118: "f32[64, 576, 7, 7]", div_24: "f32[64, 576, 7, 7]", mean_8: "f32[64, 576, 1, 1]", relu_13: "f32[64, 144, 1, 1]", div_25: "f32[64, 576, 1, 1]", mul_249: "f32[64, 576, 7, 7]", convolution_50: "f32[64, 96, 7, 7]", squeeze_97: "f32[96]", add_196: "f32[64, 96, 7, 7]", convolution_51: "f32[64, 576, 7, 7]", squeeze_100: "f32[576]", clone_119: "f32[64, 576, 7, 7]", view: "f32[64, 576]", addmm: "f32[64, 1024]", philox_seed_like: "i32[]", mul_267: "f32[64, 1024]", permute_2: "f32[1000, 1024]", permute_6: "f32[1024, 576]", unsqueeze_138: "f32[1, 576, 1, 1]", unsqueeze_150: "f32[1, 96, 1, 1]", bitwise_and: "b8[64, 576, 1, 1]", unsqueeze_162: "f32[1, 576, 1, 1]", unsqueeze_174: "f32[1, 576, 1, 1]", unsqueeze_186: "f32[1, 96, 1, 1]", bitwise_and_1: "b8[64, 576, 1, 1]", unsqueeze_198: "f32[1, 576, 1, 1]", unsqueeze_210: "f32[1, 576, 1, 1]", unsqueeze_222: "f32[1, 96, 1, 1]", bitwise_and_2: "b8[64, 288, 1, 1]", unsqueeze_234: "f32[1, 288, 1, 1]", unsqueeze_246: "f32[1, 288, 1, 1]", unsqueeze_258: "f32[1, 48, 1, 1]", bitwise_and_3: "b8[64, 144, 1, 1]", unsqueeze_270: "f32[1, 144, 1, 1]", unsqueeze_282: "f32[1, 144, 1, 1]", unsqueeze_294: "f32[1, 48, 1, 1]", bitwise_and_4: "b8[64, 120, 1, 1]", unsqueeze_306: "f32[1, 120, 1, 1]", unsqueeze_318: "f32[1, 120, 1, 1]", unsqueeze_330: "f32[1, 40, 1, 1]", bitwise_and_5: "b8[64, 240, 1, 1]", unsqueeze_342: "f32[1, 240, 1, 1]", unsqueeze_354: "f32[1, 240, 1, 1]", unsqueeze_366: "f32[1, 40, 1, 1]", bitwise_and_6: "b8[64, 240, 1, 1]", unsqueeze_378: "f32[1, 240, 1, 1]", unsqueeze_390: "f32[1, 240, 1, 1]", unsqueeze_402: "f32[1, 40, 1, 1]", bitwise_and_7: "b8[64, 96, 1, 1]", unsqueeze_414: "f32[1, 96, 1, 1]", unsqueeze_426: "f32[1, 96, 1, 1]", unsqueeze_438: "f32[1, 24, 1, 1]", unsqueeze_450: "f32[1, 88, 1, 1]", unsqueeze_462: "f32[1, 88, 1, 1]", unsqueeze_474: "f32[1, 24, 1, 1]", unsqueeze_486: "f32[1, 72, 1, 1]", unsqueeze_498: "f32[1, 72, 1, 1]", unsqueeze_510: "f32[1, 16, 1, 1]", bitwise_and_8: "b8[64, 16, 1, 1]", unsqueeze_522: "f32[1, 16, 1, 1]", unsqueeze_534: "f32[1, 16, 1, 1]", tangents_1: "f32[16]", tangents_2: "f32[16]", tangents_3: "i64[]", tangents_4: "f32[16]", tangents_5: "f32[16]", tangents_6: "i64[]", tangents_7: "f32[16]", tangents_8: "f32[16]", tangents_9: "i64[]", tangents_10: "f32[72]", tangents_11: "f32[72]", tangents_12: "i64[]", tangents_13: "f32[72]", tangents_14: "f32[72]", tangents_15: "i64[]", tangents_16: "f32[24]", tangents_17: "f32[24]", tangents_18: "i64[]", tangents_19: "f32[88]", tangents_20: "f32[88]", tangents_21: "i64[]", tangents_22: "f32[88]", tangents_23: "f32[88]", tangents_24: "i64[]", tangents_25: "f32[24]", tangents_26: "f32[24]", tangents_27: "i64[]", tangents_28: "f32[96]", tangents_29: "f32[96]", tangents_30: "i64[]", tangents_31: "f32[96]", tangents_32: "f32[96]", tangents_33: "i64[]", tangents_34: "f32[40]", tangents_35: "f32[40]", tangents_36: "i64[]", tangents_37: "f32[240]", tangents_38: "f32[240]", tangents_39: "i64[]", tangents_40: "f32[240]", tangents_41: "f32[240]", tangents_42: "i64[]", tangents_43: "f32[40]", tangents_44: "f32[40]", tangents_45: "i64[]", tangents_46: "f32[240]", tangents_47: "f32[240]", tangents_48: "i64[]", tangents_49: "f32[240]", tangents_50: "f32[240]", tangents_51: "i64[]", tangents_52: "f32[40]", tangents_53: "f32[40]", tangents_54: "i64[]", tangents_55: "f32[120]", tangents_56: "f32[120]", tangents_57: "i64[]", tangents_58: "f32[120]", tangents_59: "f32[120]", tangents_60: "i64[]", tangents_61: "f32[48]", tangents_62: "f32[48]", tangents_63: "i64[]", tangents_64: "f32[144]", tangents_65: "f32[144]", tangents_66: "i64[]", tangents_67: "f32[144]", tangents_68: "f32[144]", tangents_69: "i64[]", tangents_70: "f32[48]", tangents_71: "f32[48]", tangents_72: "i64[]", tangents_73: "f32[288]", tangents_74: "f32[288]", tangents_75: "i64[]", tangents_76: "f32[288]", tangents_77: "f32[288]", tangents_78: "i64[]", tangents_79: "f32[96]", tangents_80: "f32[96]", tangents_81: "i64[]", tangents_82: "f32[576]", tangents_83: "f32[576]", tangents_84: "i64[]", tangents_85: "f32[576]", tangents_86: "f32[576]", tangents_87: "i64[]", tangents_88: "f32[96]", tangents_89: "f32[96]", tangents_90: "i64[]", tangents_91: "f32[576]", tangents_92: "f32[576]", tangents_93: "i64[]", tangents_94: "f32[576]", tangents_95: "f32[576]", tangents_96: "i64[]", tangents_97: "f32[96]", tangents_98: "f32[96]", tangents_99: "i64[]", tangents_100: "f32[576]", tangents_101: "f32[576]", tangents_102: "i64[]", tangents_103: "f32[64, 1000]"):
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:215, code: x = self.classifier(x)
        clone_120: "f32[64, 1024]" = torch.ops.aten.clone.default(addmm);  addmm = None
        mm: "f32[64, 1024]" = torch.ops.aten.mm.default(tangents_103, permute_2);  permute_2 = None
        permute_3: "f32[1000, 64]" = torch.ops.aten.permute.default(tangents_103, [1, 0])
        mm_1: "f32[1000, 1024]" = torch.ops.aten.mm.default(permute_3, mul_267);  permute_3 = mul_267 = None
        permute_4: "f32[1024, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_103, [0], True);  tangents_103 = None
        view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
        permute_5: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        philox_rand_like_1: "f32[64, 1024]" = torch.ops.prims.philox_rand_like.default(mm, philox_seed_like, 0);  philox_seed_like = None
        gt_1: "b8[64, 1024]" = torch.ops.aten.gt.Scalar(philox_rand_like_1, 0.2);  philox_rand_like_1 = None
        convert_element_type_1: "f32[64, 1024]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_268: "f32[64, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mm);  convert_element_type_1 = mm = None
        mul_269: "f32[64, 1024]" = torch.ops.aten.mul.Tensor(mul_268, 1.25);  mul_268 = None
        lt: "b8[64, 1024]" = torch.ops.aten.lt.Scalar(clone_120, -3)
        le: "b8[64, 1024]" = torch.ops.aten.le.Scalar(clone_120, 3)
        div_28: "f32[64, 1024]" = torch.ops.aten.div.Tensor(clone_120, 3);  clone_120 = None
        add_204: "f32[64, 1024]" = torch.ops.aten.add.Tensor(div_28, 0.5);  div_28 = None
        mul_270: "f32[64, 1024]" = torch.ops.aten.mul.Tensor(mul_269, add_204);  add_204 = None
        where: "f32[64, 1024]" = torch.ops.aten.where.self(le, mul_270, mul_269);  le = mul_270 = mul_269 = None
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_1: "f32[64, 1024]" = torch.ops.aten.where.self(lt, scalar_tensor, where);  lt = where = None
        mm_2: "f32[64, 576]" = torch.ops.aten.mm.default(where_1, permute_6);  permute_6 = None
        permute_7: "f32[1024, 64]" = torch.ops.aten.permute.default(where_1, [1, 0])
        mm_3: "f32[1024, 576]" = torch.ops.aten.mm.default(permute_7, view);  permute_7 = view = None
        permute_8: "f32[576, 1024]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
        sum_2: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(where_1, [0], True);  where_1 = None
        view_2: "f32[1024]" = torch.ops.aten.view.default(sum_2, [1024]);  sum_2 = None
        permute_9: "f32[1024, 576]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:213, code: x = torch.flatten(x, 1)
        view_3: "f32[64, 576, 1, 1]" = torch.ops.aten.view.default(mm_2, [64, 576, 1, 1]);  mm_2 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:212, code: x = self.avgpool(x)
        expand: "f32[64, 576, 7, 7]" = torch.ops.aten.expand.default(view_3, [64, 576, 7, 7]);  view_3 = None
        div_29: "f32[64, 576, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:210, code: x = self.features(x)
        lt_1: "b8[64, 576, 7, 7]" = torch.ops.aten.lt.Scalar(clone_119, -3)
        le_1: "b8[64, 576, 7, 7]" = torch.ops.aten.le.Scalar(clone_119, 3)
        div_30: "f32[64, 576, 7, 7]" = torch.ops.aten.div.Tensor(clone_119, 3);  clone_119 = None
        add_205: "f32[64, 576, 7, 7]" = torch.ops.aten.add.Tensor(div_30, 0.5);  div_30 = None
        mul_271: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(div_29, add_205);  add_205 = None
        where_2: "f32[64, 576, 7, 7]" = torch.ops.aten.where.self(le_1, mul_271, div_29);  le_1 = mul_271 = div_29 = None
        where_3: "f32[64, 576, 7, 7]" = torch.ops.aten.where.self(lt_1, scalar_tensor, where_2);  lt_1 = where_2 = None
        sum_3: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        sub_34: "f32[64, 576, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_138);  convolution_51 = unsqueeze_138 = None
        mul_272: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_34)
        sum_4: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_272, [0, 2, 3]);  mul_272 = None
        mul_273: "f32[576]" = torch.ops.aten.mul.Tensor(sum_3, 0.00031887755102040814)
        unsqueeze_139: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_273, 0);  mul_273 = None
        unsqueeze_140: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 2);  unsqueeze_139 = None
        unsqueeze_141: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 3);  unsqueeze_140 = None
        mul_274: "f32[576]" = torch.ops.aten.mul.Tensor(sum_4, 0.00031887755102040814)
        mul_275: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_276: "f32[576]" = torch.ops.aten.mul.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
        unsqueeze_142: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_276, 0);  mul_276 = None
        unsqueeze_143: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 2);  unsqueeze_142 = None
        unsqueeze_144: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 3);  unsqueeze_143 = None
        mul_277: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_137);  primals_137 = None
        unsqueeze_145: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_277, 0);  mul_277 = None
        unsqueeze_146: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 2);  unsqueeze_145 = None
        unsqueeze_147: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 3);  unsqueeze_146 = None
        mul_278: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_144);  sub_34 = unsqueeze_144 = None
        sub_36: "f32[64, 576, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_278);  where_3 = mul_278 = None
        sub_37: "f32[64, 576, 7, 7]" = torch.ops.aten.sub.Tensor(sub_36, unsqueeze_141);  sub_36 = unsqueeze_141 = None
        mul_279: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_147);  sub_37 = unsqueeze_147 = None
        mul_280: "f32[576]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_100);  sum_4 = squeeze_100 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_279, add_196, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_279 = add_196 = primals_136 = None
        getitem_68: "f32[64, 96, 7, 7]" = convolution_backward[0]
        getitem_69: "f32[576, 96, 1, 1]" = convolution_backward[1];  convolution_backward = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        sum_5: "f32[96]" = torch.ops.aten.sum.dim_IntList(getitem_68, [0, 2, 3])
        sub_38: "f32[64, 96, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_150);  convolution_50 = unsqueeze_150 = None
        mul_281: "f32[64, 96, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_68, sub_38)
        sum_6: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_281, [0, 2, 3]);  mul_281 = None
        mul_282: "f32[96]" = torch.ops.aten.mul.Tensor(sum_5, 0.00031887755102040814)
        unsqueeze_151: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_282, 0);  mul_282 = None
        unsqueeze_152: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 2);  unsqueeze_151 = None
        unsqueeze_153: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, 3);  unsqueeze_152 = None
        mul_283: "f32[96]" = torch.ops.aten.mul.Tensor(sum_6, 0.00031887755102040814)
        mul_284: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_285: "f32[96]" = torch.ops.aten.mul.Tensor(mul_283, mul_284);  mul_283 = mul_284 = None
        unsqueeze_154: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_285, 0);  mul_285 = None
        unsqueeze_155: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 2);  unsqueeze_154 = None
        unsqueeze_156: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 3);  unsqueeze_155 = None
        mul_286: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_134);  primals_134 = None
        unsqueeze_157: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_286, 0);  mul_286 = None
        unsqueeze_158: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None
        unsqueeze_159: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 3);  unsqueeze_158 = None
        mul_287: "f32[64, 96, 7, 7]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_156);  sub_38 = unsqueeze_156 = None
        sub_40: "f32[64, 96, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_68, mul_287);  mul_287 = None
        sub_41: "f32[64, 96, 7, 7]" = torch.ops.aten.sub.Tensor(sub_40, unsqueeze_153);  sub_40 = unsqueeze_153 = None
        mul_288: "f32[64, 96, 7, 7]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_159);  sub_41 = unsqueeze_159 = None
        mul_289: "f32[96]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_97);  sum_6 = squeeze_97 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_288, mul_249, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_288 = mul_249 = primals_133 = None
        getitem_71: "f32[64, 576, 7, 7]" = convolution_backward_1[0]
        getitem_72: "f32[96, 576, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:260, code: return scale * input
        mul_290: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_71, div_25);  div_25 = None
        mul_291: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_71, div_24);  getitem_71 = div_24 = None
        sum_7: "f32[64, 576, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_291, [2, 3], True);  mul_291 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        mul_292: "f32[64, 576, 1, 1]" = torch.ops.aten.mul.Tensor(sum_7, 0.16666666666666666);  sum_7 = None
        where_4: "f32[64, 576, 1, 1]" = torch.ops.aten.where.self(bitwise_and, mul_292, scalar_tensor);  bitwise_and = mul_292 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
        sum_8: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(where_4, relu_13, primals_131, [576], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_4 = primals_131 = None
        getitem_74: "f32[64, 144, 1, 1]" = convolution_backward_2[0]
        getitem_75: "f32[576, 144, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
        le_2: "b8[64, 144, 1, 1]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_5: "f32[64, 144, 1, 1]" = torch.ops.aten.where.self(le_2, scalar_tensor, getitem_74);  le_2 = getitem_74 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
        sum_9: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_5, mean_8, primals_129, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = mean_8 = primals_129 = None
        getitem_77: "f32[64, 576, 1, 1]" = convolution_backward_3[0]
        getitem_78: "f32[144, 576, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        expand_1: "f32[64, 576, 7, 7]" = torch.ops.aten.expand.default(getitem_77, [64, 576, 7, 7]);  getitem_77 = None
        div_31: "f32[64, 576, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        add_206: "f32[64, 576, 7, 7]" = torch.ops.aten.add.Tensor(mul_290, div_31);  mul_290 = div_31 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        lt_3: "b8[64, 576, 7, 7]" = torch.ops.aten.lt.Scalar(clone_118, -3)
        le_3: "b8[64, 576, 7, 7]" = torch.ops.aten.le.Scalar(clone_118, 3)
        div_32: "f32[64, 576, 7, 7]" = torch.ops.aten.div.Tensor(clone_118, 3);  clone_118 = None
        add_207: "f32[64, 576, 7, 7]" = torch.ops.aten.add.Tensor(div_32, 0.5);  div_32 = None
        mul_293: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(add_206, add_207);  add_207 = None
        where_6: "f32[64, 576, 7, 7]" = torch.ops.aten.where.self(le_3, mul_293, add_206);  le_3 = mul_293 = add_206 = None
        where_7: "f32[64, 576, 7, 7]" = torch.ops.aten.where.self(lt_3, scalar_tensor, where_6);  lt_3 = where_6 = None
        sum_10: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
        sub_42: "f32[64, 576, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_162);  convolution_47 = unsqueeze_162 = None
        mul_294: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_42)
        sum_11: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_294, [0, 2, 3]);  mul_294 = None
        mul_295: "f32[576]" = torch.ops.aten.mul.Tensor(sum_10, 0.00031887755102040814)
        unsqueeze_163: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_295, 0);  mul_295 = None
        unsqueeze_164: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
        unsqueeze_165: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 3);  unsqueeze_164 = None
        mul_296: "f32[576]" = torch.ops.aten.mul.Tensor(sum_11, 0.00031887755102040814)
        mul_297: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_298: "f32[576]" = torch.ops.aten.mul.Tensor(mul_296, mul_297);  mul_296 = mul_297 = None
        unsqueeze_166: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_298, 0);  mul_298 = None
        unsqueeze_167: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 2);  unsqueeze_166 = None
        unsqueeze_168: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 3);  unsqueeze_167 = None
        mul_299: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_127);  primals_127 = None
        unsqueeze_169: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_299, 0);  mul_299 = None
        unsqueeze_170: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
        unsqueeze_171: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 3);  unsqueeze_170 = None
        mul_300: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_168);  sub_42 = unsqueeze_168 = None
        sub_44: "f32[64, 576, 7, 7]" = torch.ops.aten.sub.Tensor(where_7, mul_300);  where_7 = mul_300 = None
        sub_45: "f32[64, 576, 7, 7]" = torch.ops.aten.sub.Tensor(sub_44, unsqueeze_165);  sub_44 = unsqueeze_165 = None
        mul_301: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_171);  sub_45 = unsqueeze_171 = None
        mul_302: "f32[576]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_94);  sum_11 = squeeze_94 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_301, div_23, primals_126, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 576, [True, True, False]);  mul_301 = div_23 = primals_126 = None
        getitem_80: "f32[64, 576, 7, 7]" = convolution_backward_4[0]
        getitem_81: "f32[576, 1, 5, 5]" = convolution_backward_4[1];  convolution_backward_4 = None
        lt_4: "b8[64, 576, 7, 7]" = torch.ops.aten.lt.Scalar(clone_117, -3)
        le_4: "b8[64, 576, 7, 7]" = torch.ops.aten.le.Scalar(clone_117, 3)
        div_33: "f32[64, 576, 7, 7]" = torch.ops.aten.div.Tensor(clone_117, 3);  clone_117 = None
        add_208: "f32[64, 576, 7, 7]" = torch.ops.aten.add.Tensor(div_33, 0.5);  div_33 = None
        mul_303: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_80, add_208);  add_208 = None
        where_8: "f32[64, 576, 7, 7]" = torch.ops.aten.where.self(le_4, mul_303, getitem_80);  le_4 = mul_303 = getitem_80 = None
        where_9: "f32[64, 576, 7, 7]" = torch.ops.aten.where.self(lt_4, scalar_tensor, where_8);  lt_4 = where_8 = None
        sum_12: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
        sub_46: "f32[64, 576, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_174);  convolution_46 = unsqueeze_174 = None
        mul_304: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_46)
        sum_13: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_304, [0, 2, 3]);  mul_304 = None
        mul_305: "f32[576]" = torch.ops.aten.mul.Tensor(sum_12, 0.00031887755102040814)
        unsqueeze_175: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_305, 0);  mul_305 = None
        unsqueeze_176: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 2);  unsqueeze_175 = None
        unsqueeze_177: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 3);  unsqueeze_176 = None
        mul_306: "f32[576]" = torch.ops.aten.mul.Tensor(sum_13, 0.00031887755102040814)
        mul_307: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_308: "f32[576]" = torch.ops.aten.mul.Tensor(mul_306, mul_307);  mul_306 = mul_307 = None
        unsqueeze_178: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_308, 0);  mul_308 = None
        unsqueeze_179: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 2);  unsqueeze_178 = None
        unsqueeze_180: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 3);  unsqueeze_179 = None
        mul_309: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_124);  primals_124 = None
        unsqueeze_181: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_309, 0);  mul_309 = None
        unsqueeze_182: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 2);  unsqueeze_181 = None
        unsqueeze_183: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 3);  unsqueeze_182 = None
        mul_310: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_180);  sub_46 = unsqueeze_180 = None
        sub_48: "f32[64, 576, 7, 7]" = torch.ops.aten.sub.Tensor(where_9, mul_310);  where_9 = mul_310 = None
        sub_49: "f32[64, 576, 7, 7]" = torch.ops.aten.sub.Tensor(sub_48, unsqueeze_177);  sub_48 = unsqueeze_177 = None
        mul_311: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_183);  sub_49 = unsqueeze_183 = None
        mul_312: "f32[576]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_91);  sum_13 = squeeze_91 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_311, add_177, primals_123, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_311 = add_177 = primals_123 = None
        getitem_83: "f32[64, 96, 7, 7]" = convolution_backward_5[0]
        getitem_84: "f32[576, 96, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        add_209: "f32[64, 96, 7, 7]" = torch.ops.aten.add.Tensor(getitem_68, getitem_83);  getitem_68 = getitem_83 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        sum_14: "f32[96]" = torch.ops.aten.sum.dim_IntList(add_209, [0, 2, 3])
        sub_50: "f32[64, 96, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_186);  convolution_45 = unsqueeze_186 = None
        mul_313: "f32[64, 96, 7, 7]" = torch.ops.aten.mul.Tensor(add_209, sub_50)
        sum_15: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_313, [0, 2, 3]);  mul_313 = None
        mul_314: "f32[96]" = torch.ops.aten.mul.Tensor(sum_14, 0.00031887755102040814)
        unsqueeze_187: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_314, 0);  mul_314 = None
        unsqueeze_188: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 2);  unsqueeze_187 = None
        unsqueeze_189: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 3);  unsqueeze_188 = None
        mul_315: "f32[96]" = torch.ops.aten.mul.Tensor(sum_15, 0.00031887755102040814)
        mul_316: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_317: "f32[96]" = torch.ops.aten.mul.Tensor(mul_315, mul_316);  mul_315 = mul_316 = None
        unsqueeze_190: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_317, 0);  mul_317 = None
        unsqueeze_191: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 2);  unsqueeze_190 = None
        unsqueeze_192: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 3);  unsqueeze_191 = None
        mul_318: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_121);  primals_121 = None
        unsqueeze_193: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_318, 0);  mul_318 = None
        unsqueeze_194: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 2);  unsqueeze_193 = None
        unsqueeze_195: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 3);  unsqueeze_194 = None
        mul_319: "f32[64, 96, 7, 7]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_192);  sub_50 = unsqueeze_192 = None
        sub_52: "f32[64, 96, 7, 7]" = torch.ops.aten.sub.Tensor(add_209, mul_319);  mul_319 = None
        sub_53: "f32[64, 96, 7, 7]" = torch.ops.aten.sub.Tensor(sub_52, unsqueeze_189);  sub_52 = unsqueeze_189 = None
        mul_320: "f32[64, 96, 7, 7]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_195);  sub_53 = unsqueeze_195 = None
        mul_321: "f32[96]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_88);  sum_15 = squeeze_88 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_320, mul_225, primals_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_320 = mul_225 = primals_120 = None
        getitem_86: "f32[64, 576, 7, 7]" = convolution_backward_6[0]
        getitem_87: "f32[96, 576, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:260, code: return scale * input
        mul_322: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_86, div_22);  div_22 = None
        mul_323: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_86, div_21);  getitem_86 = div_21 = None
        sum_16: "f32[64, 576, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [2, 3], True);  mul_323 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        mul_324: "f32[64, 576, 1, 1]" = torch.ops.aten.mul.Tensor(sum_16, 0.16666666666666666);  sum_16 = None
        where_10: "f32[64, 576, 1, 1]" = torch.ops.aten.where.self(bitwise_and_1, mul_324, scalar_tensor);  bitwise_and_1 = mul_324 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
        sum_17: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(where_10, relu_12, primals_118, [576], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_10 = primals_118 = None
        getitem_89: "f32[64, 144, 1, 1]" = convolution_backward_7[0]
        getitem_90: "f32[576, 144, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
        le_5: "b8[64, 144, 1, 1]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_11: "f32[64, 144, 1, 1]" = torch.ops.aten.where.self(le_5, scalar_tensor, getitem_89);  le_5 = getitem_89 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
        sum_18: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_11, mean_7, primals_116, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_11 = mean_7 = primals_116 = None
        getitem_92: "f32[64, 576, 1, 1]" = convolution_backward_8[0]
        getitem_93: "f32[144, 576, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        expand_2: "f32[64, 576, 7, 7]" = torch.ops.aten.expand.default(getitem_92, [64, 576, 7, 7]);  getitem_92 = None
        div_34: "f32[64, 576, 7, 7]" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        add_210: "f32[64, 576, 7, 7]" = torch.ops.aten.add.Tensor(mul_322, div_34);  mul_322 = div_34 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        lt_6: "b8[64, 576, 7, 7]" = torch.ops.aten.lt.Scalar(clone_116, -3)
        le_6: "b8[64, 576, 7, 7]" = torch.ops.aten.le.Scalar(clone_116, 3)
        div_35: "f32[64, 576, 7, 7]" = torch.ops.aten.div.Tensor(clone_116, 3);  clone_116 = None
        add_211: "f32[64, 576, 7, 7]" = torch.ops.aten.add.Tensor(div_35, 0.5);  div_35 = None
        mul_325: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(add_210, add_211);  add_211 = None
        where_12: "f32[64, 576, 7, 7]" = torch.ops.aten.where.self(le_6, mul_325, add_210);  le_6 = mul_325 = add_210 = None
        where_13: "f32[64, 576, 7, 7]" = torch.ops.aten.where.self(lt_6, scalar_tensor, where_12);  lt_6 = where_12 = None
        sum_19: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
        sub_54: "f32[64, 576, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_198);  convolution_42 = unsqueeze_198 = None
        mul_326: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(where_13, sub_54)
        sum_20: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_326, [0, 2, 3]);  mul_326 = None
        mul_327: "f32[576]" = torch.ops.aten.mul.Tensor(sum_19, 0.00031887755102040814)
        unsqueeze_199: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_327, 0);  mul_327 = None
        unsqueeze_200: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
        unsqueeze_201: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 3);  unsqueeze_200 = None
        mul_328: "f32[576]" = torch.ops.aten.mul.Tensor(sum_20, 0.00031887755102040814)
        mul_329: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_330: "f32[576]" = torch.ops.aten.mul.Tensor(mul_328, mul_329);  mul_328 = mul_329 = None
        unsqueeze_202: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_330, 0);  mul_330 = None
        unsqueeze_203: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 2);  unsqueeze_202 = None
        unsqueeze_204: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 3);  unsqueeze_203 = None
        mul_331: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_114);  primals_114 = None
        unsqueeze_205: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_331, 0);  mul_331 = None
        unsqueeze_206: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
        unsqueeze_207: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 3);  unsqueeze_206 = None
        mul_332: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_204);  sub_54 = unsqueeze_204 = None
        sub_56: "f32[64, 576, 7, 7]" = torch.ops.aten.sub.Tensor(where_13, mul_332);  where_13 = mul_332 = None
        sub_57: "f32[64, 576, 7, 7]" = torch.ops.aten.sub.Tensor(sub_56, unsqueeze_201);  sub_56 = unsqueeze_201 = None
        mul_333: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_207);  sub_57 = unsqueeze_207 = None
        mul_334: "f32[576]" = torch.ops.aten.mul.Tensor(sum_20, squeeze_85);  sum_20 = squeeze_85 = None
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_333, div_20, primals_113, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 576, [True, True, False]);  mul_333 = div_20 = primals_113 = None
        getitem_95: "f32[64, 576, 7, 7]" = convolution_backward_9[0]
        getitem_96: "f32[576, 1, 5, 5]" = convolution_backward_9[1];  convolution_backward_9 = None
        lt_7: "b8[64, 576, 7, 7]" = torch.ops.aten.lt.Scalar(clone_115, -3)
        le_7: "b8[64, 576, 7, 7]" = torch.ops.aten.le.Scalar(clone_115, 3)
        div_36: "f32[64, 576, 7, 7]" = torch.ops.aten.div.Tensor(clone_115, 3);  clone_115 = None
        add_212: "f32[64, 576, 7, 7]" = torch.ops.aten.add.Tensor(div_36, 0.5);  div_36 = None
        mul_335: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_95, add_212);  add_212 = None
        where_14: "f32[64, 576, 7, 7]" = torch.ops.aten.where.self(le_7, mul_335, getitem_95);  le_7 = mul_335 = getitem_95 = None
        where_15: "f32[64, 576, 7, 7]" = torch.ops.aten.where.self(lt_7, scalar_tensor, where_14);  lt_7 = where_14 = None
        sum_21: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
        sub_58: "f32[64, 576, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_210);  convolution_41 = unsqueeze_210 = None
        mul_336: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(where_15, sub_58)
        sum_22: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_336, [0, 2, 3]);  mul_336 = None
        mul_337: "f32[576]" = torch.ops.aten.mul.Tensor(sum_21, 0.00031887755102040814)
        unsqueeze_211: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_337, 0);  mul_337 = None
        unsqueeze_212: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
        unsqueeze_213: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 3);  unsqueeze_212 = None
        mul_338: "f32[576]" = torch.ops.aten.mul.Tensor(sum_22, 0.00031887755102040814)
        mul_339: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_340: "f32[576]" = torch.ops.aten.mul.Tensor(mul_338, mul_339);  mul_338 = mul_339 = None
        unsqueeze_214: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_340, 0);  mul_340 = None
        unsqueeze_215: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
        unsqueeze_216: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 3);  unsqueeze_215 = None
        mul_341: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_111);  primals_111 = None
        unsqueeze_217: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_341, 0);  mul_341 = None
        unsqueeze_218: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
        unsqueeze_219: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
        mul_342: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_216);  sub_58 = unsqueeze_216 = None
        sub_60: "f32[64, 576, 7, 7]" = torch.ops.aten.sub.Tensor(where_15, mul_342);  where_15 = mul_342 = None
        sub_61: "f32[64, 576, 7, 7]" = torch.ops.aten.sub.Tensor(sub_60, unsqueeze_213);  sub_60 = unsqueeze_213 = None
        mul_343: "f32[64, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_219);  sub_61 = unsqueeze_219 = None
        mul_344: "f32[576]" = torch.ops.aten.mul.Tensor(sum_22, squeeze_82);  sum_22 = squeeze_82 = None
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_343, add_158, primals_110, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_343 = add_158 = primals_110 = None
        getitem_98: "f32[64, 96, 7, 7]" = convolution_backward_10[0]
        getitem_99: "f32[576, 96, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        add_213: "f32[64, 96, 7, 7]" = torch.ops.aten.add.Tensor(add_209, getitem_98);  add_209 = getitem_98 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        sum_23: "f32[96]" = torch.ops.aten.sum.dim_IntList(add_213, [0, 2, 3])
        sub_62: "f32[64, 96, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_222);  convolution_40 = unsqueeze_222 = None
        mul_345: "f32[64, 96, 7, 7]" = torch.ops.aten.mul.Tensor(add_213, sub_62)
        sum_24: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_345, [0, 2, 3]);  mul_345 = None
        mul_346: "f32[96]" = torch.ops.aten.mul.Tensor(sum_23, 0.00031887755102040814)
        unsqueeze_223: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_346, 0);  mul_346 = None
        unsqueeze_224: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
        unsqueeze_225: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
        mul_347: "f32[96]" = torch.ops.aten.mul.Tensor(sum_24, 0.00031887755102040814)
        mul_348: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_349: "f32[96]" = torch.ops.aten.mul.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
        unsqueeze_226: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_349, 0);  mul_349 = None
        unsqueeze_227: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
        unsqueeze_228: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
        mul_350: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_108);  primals_108 = None
        unsqueeze_229: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_350, 0);  mul_350 = None
        unsqueeze_230: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
        unsqueeze_231: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
        mul_351: "f32[64, 96, 7, 7]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_228);  sub_62 = unsqueeze_228 = None
        sub_64: "f32[64, 96, 7, 7]" = torch.ops.aten.sub.Tensor(add_213, mul_351);  add_213 = mul_351 = None
        sub_65: "f32[64, 96, 7, 7]" = torch.ops.aten.sub.Tensor(sub_64, unsqueeze_225);  sub_64 = unsqueeze_225 = None
        mul_352: "f32[64, 96, 7, 7]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_231);  sub_65 = unsqueeze_231 = None
        mul_353: "f32[96]" = torch.ops.aten.mul.Tensor(sum_24, squeeze_79);  sum_24 = squeeze_79 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_352, mul_201, primals_107, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_352 = mul_201 = primals_107 = None
        getitem_101: "f32[64, 288, 7, 7]" = convolution_backward_11[0]
        getitem_102: "f32[96, 288, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:260, code: return scale * input
        mul_354: "f32[64, 288, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_101, div_19);  div_19 = None
        mul_355: "f32[64, 288, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_101, div_18);  getitem_101 = div_18 = None
        sum_25: "f32[64, 288, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_355, [2, 3], True);  mul_355 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        mul_356: "f32[64, 288, 1, 1]" = torch.ops.aten.mul.Tensor(sum_25, 0.16666666666666666);  sum_25 = None
        where_16: "f32[64, 288, 1, 1]" = torch.ops.aten.where.self(bitwise_and_2, mul_356, scalar_tensor);  bitwise_and_2 = mul_356 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
        sum_26: "f32[288]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(where_16, relu_11, primals_105, [288], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_16 = primals_105 = None
        getitem_104: "f32[64, 72, 1, 1]" = convolution_backward_12[0]
        getitem_105: "f32[288, 72, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
        le_8: "b8[64, 72, 1, 1]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_17: "f32[64, 72, 1, 1]" = torch.ops.aten.where.self(le_8, scalar_tensor, getitem_104);  le_8 = getitem_104 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
        sum_27: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(where_17, mean_6, primals_103, [72], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_17 = mean_6 = primals_103 = None
        getitem_107: "f32[64, 288, 1, 1]" = convolution_backward_13[0]
        getitem_108: "f32[72, 288, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        expand_3: "f32[64, 288, 7, 7]" = torch.ops.aten.expand.default(getitem_107, [64, 288, 7, 7]);  getitem_107 = None
        div_37: "f32[64, 288, 7, 7]" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        add_214: "f32[64, 288, 7, 7]" = torch.ops.aten.add.Tensor(mul_354, div_37);  mul_354 = div_37 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        lt_9: "b8[64, 288, 7, 7]" = torch.ops.aten.lt.Scalar(clone_114, -3)
        le_9: "b8[64, 288, 7, 7]" = torch.ops.aten.le.Scalar(clone_114, 3)
        div_38: "f32[64, 288, 7, 7]" = torch.ops.aten.div.Tensor(clone_114, 3);  clone_114 = None
        add_215: "f32[64, 288, 7, 7]" = torch.ops.aten.add.Tensor(div_38, 0.5);  div_38 = None
        mul_357: "f32[64, 288, 7, 7]" = torch.ops.aten.mul.Tensor(add_214, add_215);  add_215 = None
        where_18: "f32[64, 288, 7, 7]" = torch.ops.aten.where.self(le_9, mul_357, add_214);  le_9 = mul_357 = add_214 = None
        where_19: "f32[64, 288, 7, 7]" = torch.ops.aten.where.self(lt_9, scalar_tensor, where_18);  lt_9 = where_18 = None
        sum_28: "f32[288]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
        sub_66: "f32[64, 288, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_234);  convolution_37 = unsqueeze_234 = None
        mul_358: "f32[64, 288, 7, 7]" = torch.ops.aten.mul.Tensor(where_19, sub_66)
        sum_29: "f32[288]" = torch.ops.aten.sum.dim_IntList(mul_358, [0, 2, 3]);  mul_358 = None
        mul_359: "f32[288]" = torch.ops.aten.mul.Tensor(sum_28, 0.00031887755102040814)
        unsqueeze_235: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_359, 0);  mul_359 = None
        unsqueeze_236: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
        unsqueeze_237: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
        mul_360: "f32[288]" = torch.ops.aten.mul.Tensor(sum_29, 0.00031887755102040814)
        mul_361: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_362: "f32[288]" = torch.ops.aten.mul.Tensor(mul_360, mul_361);  mul_360 = mul_361 = None
        unsqueeze_238: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_362, 0);  mul_362 = None
        unsqueeze_239: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
        unsqueeze_240: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
        mul_363: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_101);  primals_101 = None
        unsqueeze_241: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_363, 0);  mul_363 = None
        unsqueeze_242: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
        unsqueeze_243: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
        mul_364: "f32[64, 288, 7, 7]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_240);  sub_66 = unsqueeze_240 = None
        sub_68: "f32[64, 288, 7, 7]" = torch.ops.aten.sub.Tensor(where_19, mul_364);  where_19 = mul_364 = None
        sub_69: "f32[64, 288, 7, 7]" = torch.ops.aten.sub.Tensor(sub_68, unsqueeze_237);  sub_68 = unsqueeze_237 = None
        mul_365: "f32[64, 288, 7, 7]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_243);  sub_69 = unsqueeze_243 = None
        mul_366: "f32[288]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_76);  sum_29 = squeeze_76 = None
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_365, div_17, primals_100, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 288, [True, True, False]);  mul_365 = div_17 = primals_100 = None
        getitem_110: "f32[64, 288, 14, 14]" = convolution_backward_14[0]
        getitem_111: "f32[288, 1, 5, 5]" = convolution_backward_14[1];  convolution_backward_14 = None
        lt_10: "b8[64, 288, 14, 14]" = torch.ops.aten.lt.Scalar(clone_113, -3)
        le_10: "b8[64, 288, 14, 14]" = torch.ops.aten.le.Scalar(clone_113, 3)
        div_39: "f32[64, 288, 14, 14]" = torch.ops.aten.div.Tensor(clone_113, 3);  clone_113 = None
        add_216: "f32[64, 288, 14, 14]" = torch.ops.aten.add.Tensor(div_39, 0.5);  div_39 = None
        mul_367: "f32[64, 288, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_110, add_216);  add_216 = None
        where_20: "f32[64, 288, 14, 14]" = torch.ops.aten.where.self(le_10, mul_367, getitem_110);  le_10 = mul_367 = getitem_110 = None
        where_21: "f32[64, 288, 14, 14]" = torch.ops.aten.where.self(lt_10, scalar_tensor, where_20);  lt_10 = where_20 = None
        sum_30: "f32[288]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
        sub_70: "f32[64, 288, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_246);  convolution_36 = unsqueeze_246 = None
        mul_368: "f32[64, 288, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_70)
        sum_31: "f32[288]" = torch.ops.aten.sum.dim_IntList(mul_368, [0, 2, 3]);  mul_368 = None
        mul_369: "f32[288]" = torch.ops.aten.mul.Tensor(sum_30, 7.971938775510203e-05)
        unsqueeze_247: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_369, 0);  mul_369 = None
        unsqueeze_248: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
        unsqueeze_249: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
        mul_370: "f32[288]" = torch.ops.aten.mul.Tensor(sum_31, 7.971938775510203e-05)
        mul_371: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_372: "f32[288]" = torch.ops.aten.mul.Tensor(mul_370, mul_371);  mul_370 = mul_371 = None
        unsqueeze_250: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_372, 0);  mul_372 = None
        unsqueeze_251: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
        unsqueeze_252: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
        mul_373: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_98);  primals_98 = None
        unsqueeze_253: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_373, 0);  mul_373 = None
        unsqueeze_254: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
        unsqueeze_255: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
        mul_374: "f32[64, 288, 14, 14]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_252);  sub_70 = unsqueeze_252 = None
        sub_72: "f32[64, 288, 14, 14]" = torch.ops.aten.sub.Tensor(where_21, mul_374);  where_21 = mul_374 = None
        sub_73: "f32[64, 288, 14, 14]" = torch.ops.aten.sub.Tensor(sub_72, unsqueeze_249);  sub_72 = unsqueeze_249 = None
        mul_375: "f32[64, 288, 14, 14]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_255);  sub_73 = unsqueeze_255 = None
        mul_376: "f32[288]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_73);  sum_31 = squeeze_73 = None
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_375, add_140, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_375 = add_140 = primals_97 = None
        getitem_113: "f32[64, 48, 14, 14]" = convolution_backward_15[0]
        getitem_114: "f32[288, 48, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
        sum_32: "f32[48]" = torch.ops.aten.sum.dim_IntList(getitem_113, [0, 2, 3])
        sub_74: "f32[64, 48, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_258);  convolution_35 = unsqueeze_258 = None
        mul_377: "f32[64, 48, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_113, sub_74)
        sum_33: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_377, [0, 2, 3]);  mul_377 = None
        mul_378: "f32[48]" = torch.ops.aten.mul.Tensor(sum_32, 7.971938775510203e-05)
        unsqueeze_259: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_378, 0);  mul_378 = None
        unsqueeze_260: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
        unsqueeze_261: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
        mul_379: "f32[48]" = torch.ops.aten.mul.Tensor(sum_33, 7.971938775510203e-05)
        mul_380: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_381: "f32[48]" = torch.ops.aten.mul.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
        unsqueeze_262: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_381, 0);  mul_381 = None
        unsqueeze_263: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
        unsqueeze_264: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
        mul_382: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_95);  primals_95 = None
        unsqueeze_265: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_382, 0);  mul_382 = None
        unsqueeze_266: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
        unsqueeze_267: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
        mul_383: "f32[64, 48, 14, 14]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_264);  sub_74 = unsqueeze_264 = None
        sub_76: "f32[64, 48, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_113, mul_383);  mul_383 = None
        sub_77: "f32[64, 48, 14, 14]" = torch.ops.aten.sub.Tensor(sub_76, unsqueeze_261);  sub_76 = unsqueeze_261 = None
        mul_384: "f32[64, 48, 14, 14]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_267);  sub_77 = unsqueeze_267 = None
        mul_385: "f32[48]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_70);  sum_33 = squeeze_70 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_384, mul_177, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_384 = mul_177 = primals_94 = None
        getitem_116: "f32[64, 144, 14, 14]" = convolution_backward_16[0]
        getitem_117: "f32[48, 144, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:260, code: return scale * input
        mul_386: "f32[64, 144, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_116, div_16);  div_16 = None
        mul_387: "f32[64, 144, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_116, div_15);  getitem_116 = div_15 = None
        sum_34: "f32[64, 144, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_387, [2, 3], True);  mul_387 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        mul_388: "f32[64, 144, 1, 1]" = torch.ops.aten.mul.Tensor(sum_34, 0.16666666666666666);  sum_34 = None
        where_22: "f32[64, 144, 1, 1]" = torch.ops.aten.where.self(bitwise_and_3, mul_388, scalar_tensor);  bitwise_and_3 = mul_388 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
        sum_35: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(where_22, relu_10, primals_92, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_22 = primals_92 = None
        getitem_119: "f32[64, 40, 1, 1]" = convolution_backward_17[0]
        getitem_120: "f32[144, 40, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
        le_11: "b8[64, 40, 1, 1]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_23: "f32[64, 40, 1, 1]" = torch.ops.aten.where.self(le_11, scalar_tensor, getitem_119);  le_11 = getitem_119 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
        sum_36: "f32[40]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(where_23, mean_5, primals_90, [40], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_23 = mean_5 = primals_90 = None
        getitem_122: "f32[64, 144, 1, 1]" = convolution_backward_18[0]
        getitem_123: "f32[40, 144, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        expand_4: "f32[64, 144, 14, 14]" = torch.ops.aten.expand.default(getitem_122, [64, 144, 14, 14]);  getitem_122 = None
        div_40: "f32[64, 144, 14, 14]" = torch.ops.aten.div.Scalar(expand_4, 196);  expand_4 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        add_217: "f32[64, 144, 14, 14]" = torch.ops.aten.add.Tensor(mul_386, div_40);  mul_386 = div_40 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        lt_12: "b8[64, 144, 14, 14]" = torch.ops.aten.lt.Scalar(clone_112, -3)
        le_12: "b8[64, 144, 14, 14]" = torch.ops.aten.le.Scalar(clone_112, 3)
        div_41: "f32[64, 144, 14, 14]" = torch.ops.aten.div.Tensor(clone_112, 3);  clone_112 = None
        add_218: "f32[64, 144, 14, 14]" = torch.ops.aten.add.Tensor(div_41, 0.5);  div_41 = None
        mul_389: "f32[64, 144, 14, 14]" = torch.ops.aten.mul.Tensor(add_217, add_218);  add_218 = None
        where_24: "f32[64, 144, 14, 14]" = torch.ops.aten.where.self(le_12, mul_389, add_217);  le_12 = mul_389 = add_217 = None
        where_25: "f32[64, 144, 14, 14]" = torch.ops.aten.where.self(lt_12, scalar_tensor, where_24);  lt_12 = where_24 = None
        sum_37: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
        sub_78: "f32[64, 144, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_270);  convolution_32 = unsqueeze_270 = None
        mul_390: "f32[64, 144, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, sub_78)
        sum_38: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_390, [0, 2, 3]);  mul_390 = None
        mul_391: "f32[144]" = torch.ops.aten.mul.Tensor(sum_37, 7.971938775510203e-05)
        unsqueeze_271: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_391, 0);  mul_391 = None
        unsqueeze_272: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
        unsqueeze_273: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
        mul_392: "f32[144]" = torch.ops.aten.mul.Tensor(sum_38, 7.971938775510203e-05)
        mul_393: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_394: "f32[144]" = torch.ops.aten.mul.Tensor(mul_392, mul_393);  mul_392 = mul_393 = None
        unsqueeze_274: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_394, 0);  mul_394 = None
        unsqueeze_275: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
        unsqueeze_276: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
        mul_395: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_88);  primals_88 = None
        unsqueeze_277: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_395, 0);  mul_395 = None
        unsqueeze_278: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
        unsqueeze_279: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
        mul_396: "f32[64, 144, 14, 14]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_276);  sub_78 = unsqueeze_276 = None
        sub_80: "f32[64, 144, 14, 14]" = torch.ops.aten.sub.Tensor(where_25, mul_396);  where_25 = mul_396 = None
        sub_81: "f32[64, 144, 14, 14]" = torch.ops.aten.sub.Tensor(sub_80, unsqueeze_273);  sub_80 = unsqueeze_273 = None
        mul_397: "f32[64, 144, 14, 14]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_279);  sub_81 = unsqueeze_279 = None
        mul_398: "f32[144]" = torch.ops.aten.mul.Tensor(sum_38, squeeze_67);  sum_38 = squeeze_67 = None
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_397, div_14, primals_87, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 144, [True, True, False]);  mul_397 = div_14 = primals_87 = None
        getitem_125: "f32[64, 144, 14, 14]" = convolution_backward_19[0]
        getitem_126: "f32[144, 1, 5, 5]" = convolution_backward_19[1];  convolution_backward_19 = None
        lt_13: "b8[64, 144, 14, 14]" = torch.ops.aten.lt.Scalar(clone_111, -3)
        le_13: "b8[64, 144, 14, 14]" = torch.ops.aten.le.Scalar(clone_111, 3)
        div_42: "f32[64, 144, 14, 14]" = torch.ops.aten.div.Tensor(clone_111, 3);  clone_111 = None
        add_219: "f32[64, 144, 14, 14]" = torch.ops.aten.add.Tensor(div_42, 0.5);  div_42 = None
        mul_399: "f32[64, 144, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_125, add_219);  add_219 = None
        where_26: "f32[64, 144, 14, 14]" = torch.ops.aten.where.self(le_13, mul_399, getitem_125);  le_13 = mul_399 = getitem_125 = None
        where_27: "f32[64, 144, 14, 14]" = torch.ops.aten.where.self(lt_13, scalar_tensor, where_26);  lt_13 = where_26 = None
        sum_39: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
        sub_82: "f32[64, 144, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_282);  convolution_31 = unsqueeze_282 = None
        mul_400: "f32[64, 144, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, sub_82)
        sum_40: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 2, 3]);  mul_400 = None
        mul_401: "f32[144]" = torch.ops.aten.mul.Tensor(sum_39, 7.971938775510203e-05)
        unsqueeze_283: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_401, 0);  mul_401 = None
        unsqueeze_284: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
        unsqueeze_285: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
        mul_402: "f32[144]" = torch.ops.aten.mul.Tensor(sum_40, 7.971938775510203e-05)
        mul_403: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_404: "f32[144]" = torch.ops.aten.mul.Tensor(mul_402, mul_403);  mul_402 = mul_403 = None
        unsqueeze_286: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_404, 0);  mul_404 = None
        unsqueeze_287: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
        unsqueeze_288: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
        mul_405: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_85);  primals_85 = None
        unsqueeze_289: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_405, 0);  mul_405 = None
        unsqueeze_290: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
        unsqueeze_291: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
        mul_406: "f32[64, 144, 14, 14]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_288);  sub_82 = unsqueeze_288 = None
        sub_84: "f32[64, 144, 14, 14]" = torch.ops.aten.sub.Tensor(where_27, mul_406);  where_27 = mul_406 = None
        sub_85: "f32[64, 144, 14, 14]" = torch.ops.aten.sub.Tensor(sub_84, unsqueeze_285);  sub_84 = unsqueeze_285 = None
        mul_407: "f32[64, 144, 14, 14]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_291);  sub_85 = unsqueeze_291 = None
        mul_408: "f32[144]" = torch.ops.aten.mul.Tensor(sum_40, squeeze_64);  sum_40 = squeeze_64 = None
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_407, add_121, primals_84, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_407 = add_121 = primals_84 = None
        getitem_128: "f32[64, 48, 14, 14]" = convolution_backward_20[0]
        getitem_129: "f32[144, 48, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        add_220: "f32[64, 48, 14, 14]" = torch.ops.aten.add.Tensor(getitem_113, getitem_128);  getitem_113 = getitem_128 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        sum_41: "f32[48]" = torch.ops.aten.sum.dim_IntList(add_220, [0, 2, 3])
        sub_86: "f32[64, 48, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_294);  convolution_30 = unsqueeze_294 = None
        mul_409: "f32[64, 48, 14, 14]" = torch.ops.aten.mul.Tensor(add_220, sub_86)
        sum_42: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 2, 3]);  mul_409 = None
        mul_410: "f32[48]" = torch.ops.aten.mul.Tensor(sum_41, 7.971938775510203e-05)
        unsqueeze_295: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_410, 0);  mul_410 = None
        unsqueeze_296: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
        unsqueeze_297: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
        mul_411: "f32[48]" = torch.ops.aten.mul.Tensor(sum_42, 7.971938775510203e-05)
        mul_412: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_413: "f32[48]" = torch.ops.aten.mul.Tensor(mul_411, mul_412);  mul_411 = mul_412 = None
        unsqueeze_298: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_413, 0);  mul_413 = None
        unsqueeze_299: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
        unsqueeze_300: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
        mul_414: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_82);  primals_82 = None
        unsqueeze_301: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_414, 0);  mul_414 = None
        unsqueeze_302: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
        unsqueeze_303: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
        mul_415: "f32[64, 48, 14, 14]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_300);  sub_86 = unsqueeze_300 = None
        sub_88: "f32[64, 48, 14, 14]" = torch.ops.aten.sub.Tensor(add_220, mul_415);  add_220 = mul_415 = None
        sub_89: "f32[64, 48, 14, 14]" = torch.ops.aten.sub.Tensor(sub_88, unsqueeze_297);  sub_88 = unsqueeze_297 = None
        mul_416: "f32[64, 48, 14, 14]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_303);  sub_89 = unsqueeze_303 = None
        mul_417: "f32[48]" = torch.ops.aten.mul.Tensor(sum_42, squeeze_61);  sum_42 = squeeze_61 = None
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_416, mul_153, primals_81, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_416 = mul_153 = primals_81 = None
        getitem_131: "f32[64, 120, 14, 14]" = convolution_backward_21[0]
        getitem_132: "f32[48, 120, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:260, code: return scale * input
        mul_418: "f32[64, 120, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_131, div_13);  div_13 = None
        mul_419: "f32[64, 120, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_131, div_12);  getitem_131 = div_12 = None
        sum_43: "f32[64, 120, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_419, [2, 3], True);  mul_419 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        mul_420: "f32[64, 120, 1, 1]" = torch.ops.aten.mul.Tensor(sum_43, 0.16666666666666666);  sum_43 = None
        where_28: "f32[64, 120, 1, 1]" = torch.ops.aten.where.self(bitwise_and_4, mul_420, scalar_tensor);  bitwise_and_4 = mul_420 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
        sum_44: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(where_28, relu_9, primals_79, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_28 = primals_79 = None
        getitem_134: "f32[64, 32, 1, 1]" = convolution_backward_22[0]
        getitem_135: "f32[120, 32, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
        le_14: "b8[64, 32, 1, 1]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_29: "f32[64, 32, 1, 1]" = torch.ops.aten.where.self(le_14, scalar_tensor, getitem_134);  le_14 = getitem_134 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
        sum_45: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(where_29, mean_4, primals_77, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_29 = mean_4 = primals_77 = None
        getitem_137: "f32[64, 120, 1, 1]" = convolution_backward_23[0]
        getitem_138: "f32[32, 120, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        expand_5: "f32[64, 120, 14, 14]" = torch.ops.aten.expand.default(getitem_137, [64, 120, 14, 14]);  getitem_137 = None
        div_43: "f32[64, 120, 14, 14]" = torch.ops.aten.div.Scalar(expand_5, 196);  expand_5 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        add_221: "f32[64, 120, 14, 14]" = torch.ops.aten.add.Tensor(mul_418, div_43);  mul_418 = div_43 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        lt_15: "b8[64, 120, 14, 14]" = torch.ops.aten.lt.Scalar(clone_110, -3)
        le_15: "b8[64, 120, 14, 14]" = torch.ops.aten.le.Scalar(clone_110, 3)
        div_44: "f32[64, 120, 14, 14]" = torch.ops.aten.div.Tensor(clone_110, 3);  clone_110 = None
        add_222: "f32[64, 120, 14, 14]" = torch.ops.aten.add.Tensor(div_44, 0.5);  div_44 = None
        mul_421: "f32[64, 120, 14, 14]" = torch.ops.aten.mul.Tensor(add_221, add_222);  add_222 = None
        where_30: "f32[64, 120, 14, 14]" = torch.ops.aten.where.self(le_15, mul_421, add_221);  le_15 = mul_421 = add_221 = None
        where_31: "f32[64, 120, 14, 14]" = torch.ops.aten.where.self(lt_15, scalar_tensor, where_30);  lt_15 = where_30 = None
        sum_46: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
        sub_90: "f32[64, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_306);  convolution_27 = unsqueeze_306 = None
        mul_422: "f32[64, 120, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, sub_90)
        sum_47: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_422, [0, 2, 3]);  mul_422 = None
        mul_423: "f32[120]" = torch.ops.aten.mul.Tensor(sum_46, 7.971938775510203e-05)
        unsqueeze_307: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
        unsqueeze_308: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
        unsqueeze_309: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
        mul_424: "f32[120]" = torch.ops.aten.mul.Tensor(sum_47, 7.971938775510203e-05)
        mul_425: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_426: "f32[120]" = torch.ops.aten.mul.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None
        unsqueeze_310: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_426, 0);  mul_426 = None
        unsqueeze_311: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
        unsqueeze_312: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
        mul_427: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_75);  primals_75 = None
        unsqueeze_313: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_427, 0);  mul_427 = None
        unsqueeze_314: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
        unsqueeze_315: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
        mul_428: "f32[64, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_312);  sub_90 = unsqueeze_312 = None
        sub_92: "f32[64, 120, 14, 14]" = torch.ops.aten.sub.Tensor(where_31, mul_428);  where_31 = mul_428 = None
        sub_93: "f32[64, 120, 14, 14]" = torch.ops.aten.sub.Tensor(sub_92, unsqueeze_309);  sub_92 = unsqueeze_309 = None
        mul_429: "f32[64, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_315);  sub_93 = unsqueeze_315 = None
        mul_430: "f32[120]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_58);  sum_47 = squeeze_58 = None
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_429, div_11, primals_74, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_429 = div_11 = primals_74 = None
        getitem_140: "f32[64, 120, 14, 14]" = convolution_backward_24[0]
        getitem_141: "f32[120, 1, 5, 5]" = convolution_backward_24[1];  convolution_backward_24 = None
        lt_16: "b8[64, 120, 14, 14]" = torch.ops.aten.lt.Scalar(clone_109, -3)
        le_16: "b8[64, 120, 14, 14]" = torch.ops.aten.le.Scalar(clone_109, 3)
        div_45: "f32[64, 120, 14, 14]" = torch.ops.aten.div.Tensor(clone_109, 3);  clone_109 = None
        add_223: "f32[64, 120, 14, 14]" = torch.ops.aten.add.Tensor(div_45, 0.5);  div_45 = None
        mul_431: "f32[64, 120, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_140, add_223);  add_223 = None
        where_32: "f32[64, 120, 14, 14]" = torch.ops.aten.where.self(le_16, mul_431, getitem_140);  le_16 = mul_431 = getitem_140 = None
        where_33: "f32[64, 120, 14, 14]" = torch.ops.aten.where.self(lt_16, scalar_tensor, where_32);  lt_16 = where_32 = None
        sum_48: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
        sub_94: "f32[64, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_318);  convolution_26 = unsqueeze_318 = None
        mul_432: "f32[64, 120, 14, 14]" = torch.ops.aten.mul.Tensor(where_33, sub_94)
        sum_49: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 2, 3]);  mul_432 = None
        mul_433: "f32[120]" = torch.ops.aten.mul.Tensor(sum_48, 7.971938775510203e-05)
        unsqueeze_319: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_433, 0);  mul_433 = None
        unsqueeze_320: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
        unsqueeze_321: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
        mul_434: "f32[120]" = torch.ops.aten.mul.Tensor(sum_49, 7.971938775510203e-05)
        mul_435: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_436: "f32[120]" = torch.ops.aten.mul.Tensor(mul_434, mul_435);  mul_434 = mul_435 = None
        unsqueeze_322: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_436, 0);  mul_436 = None
        unsqueeze_323: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
        unsqueeze_324: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
        mul_437: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_72);  primals_72 = None
        unsqueeze_325: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_437, 0);  mul_437 = None
        unsqueeze_326: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
        unsqueeze_327: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
        mul_438: "f32[64, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_324);  sub_94 = unsqueeze_324 = None
        sub_96: "f32[64, 120, 14, 14]" = torch.ops.aten.sub.Tensor(where_33, mul_438);  where_33 = mul_438 = None
        sub_97: "f32[64, 120, 14, 14]" = torch.ops.aten.sub.Tensor(sub_96, unsqueeze_321);  sub_96 = unsqueeze_321 = None
        mul_439: "f32[64, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_327);  sub_97 = unsqueeze_327 = None
        mul_440: "f32[120]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_55);  sum_49 = squeeze_55 = None
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_439, add_103, primals_71, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_439 = add_103 = primals_71 = None
        getitem_143: "f32[64, 40, 14, 14]" = convolution_backward_25[0]
        getitem_144: "f32[120, 40, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
        sum_50: "f32[40]" = torch.ops.aten.sum.dim_IntList(getitem_143, [0, 2, 3])
        sub_98: "f32[64, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_330);  convolution_25 = unsqueeze_330 = None
        mul_441: "f32[64, 40, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_143, sub_98)
        sum_51: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_441, [0, 2, 3]);  mul_441 = None
        mul_442: "f32[40]" = torch.ops.aten.mul.Tensor(sum_50, 7.971938775510203e-05)
        unsqueeze_331: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_442, 0);  mul_442 = None
        unsqueeze_332: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
        unsqueeze_333: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
        mul_443: "f32[40]" = torch.ops.aten.mul.Tensor(sum_51, 7.971938775510203e-05)
        mul_444: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_445: "f32[40]" = torch.ops.aten.mul.Tensor(mul_443, mul_444);  mul_443 = mul_444 = None
        unsqueeze_334: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_445, 0);  mul_445 = None
        unsqueeze_335: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
        unsqueeze_336: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
        mul_446: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_69);  primals_69 = None
        unsqueeze_337: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_446, 0);  mul_446 = None
        unsqueeze_338: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
        unsqueeze_339: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
        mul_447: "f32[64, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_336);  sub_98 = unsqueeze_336 = None
        sub_100: "f32[64, 40, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_143, mul_447);  mul_447 = None
        sub_101: "f32[64, 40, 14, 14]" = torch.ops.aten.sub.Tensor(sub_100, unsqueeze_333);  sub_100 = unsqueeze_333 = None
        mul_448: "f32[64, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_339);  sub_101 = unsqueeze_339 = None
        mul_449: "f32[40]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_52);  sum_51 = squeeze_52 = None
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_448, mul_129, primals_68, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_448 = mul_129 = primals_68 = None
        getitem_146: "f32[64, 240, 14, 14]" = convolution_backward_26[0]
        getitem_147: "f32[40, 240, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:260, code: return scale * input
        mul_450: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_146, div_10);  div_10 = None
        mul_451: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_146, div_9);  getitem_146 = div_9 = None
        sum_52: "f32[64, 240, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_451, [2, 3], True);  mul_451 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        mul_452: "f32[64, 240, 1, 1]" = torch.ops.aten.mul.Tensor(sum_52, 0.16666666666666666);  sum_52 = None
        where_34: "f32[64, 240, 1, 1]" = torch.ops.aten.where.self(bitwise_and_5, mul_452, scalar_tensor);  bitwise_and_5 = mul_452 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
        sum_53: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(where_34, relu_8, primals_66, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_34 = primals_66 = None
        getitem_149: "f32[64, 64, 1, 1]" = convolution_backward_27[0]
        getitem_150: "f32[240, 64, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
        le_17: "b8[64, 64, 1, 1]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_35: "f32[64, 64, 1, 1]" = torch.ops.aten.where.self(le_17, scalar_tensor, getitem_149);  le_17 = getitem_149 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
        sum_54: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(where_35, mean_3, primals_64, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_35 = mean_3 = primals_64 = None
        getitem_152: "f32[64, 240, 1, 1]" = convolution_backward_28[0]
        getitem_153: "f32[64, 240, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        expand_6: "f32[64, 240, 14, 14]" = torch.ops.aten.expand.default(getitem_152, [64, 240, 14, 14]);  getitem_152 = None
        div_46: "f32[64, 240, 14, 14]" = torch.ops.aten.div.Scalar(expand_6, 196);  expand_6 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        add_224: "f32[64, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_450, div_46);  mul_450 = div_46 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        lt_18: "b8[64, 240, 14, 14]" = torch.ops.aten.lt.Scalar(clone_108, -3)
        le_18: "b8[64, 240, 14, 14]" = torch.ops.aten.le.Scalar(clone_108, 3)
        div_47: "f32[64, 240, 14, 14]" = torch.ops.aten.div.Tensor(clone_108, 3);  clone_108 = None
        add_225: "f32[64, 240, 14, 14]" = torch.ops.aten.add.Tensor(div_47, 0.5);  div_47 = None
        mul_453: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(add_224, add_225);  add_225 = None
        where_36: "f32[64, 240, 14, 14]" = torch.ops.aten.where.self(le_18, mul_453, add_224);  le_18 = mul_453 = add_224 = None
        where_37: "f32[64, 240, 14, 14]" = torch.ops.aten.where.self(lt_18, scalar_tensor, where_36);  lt_18 = where_36 = None
        sum_55: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
        sub_102: "f32[64, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_342);  convolution_22 = unsqueeze_342 = None
        mul_454: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_37, sub_102)
        sum_56: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_454, [0, 2, 3]);  mul_454 = None
        mul_455: "f32[240]" = torch.ops.aten.mul.Tensor(sum_55, 7.971938775510203e-05)
        unsqueeze_343: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_455, 0);  mul_455 = None
        unsqueeze_344: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
        unsqueeze_345: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
        mul_456: "f32[240]" = torch.ops.aten.mul.Tensor(sum_56, 7.971938775510203e-05)
        mul_457: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_458: "f32[240]" = torch.ops.aten.mul.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
        unsqueeze_346: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
        unsqueeze_347: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
        unsqueeze_348: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
        mul_459: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_62);  primals_62 = None
        unsqueeze_349: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
        unsqueeze_350: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
        unsqueeze_351: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
        mul_460: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_348);  sub_102 = unsqueeze_348 = None
        sub_104: "f32[64, 240, 14, 14]" = torch.ops.aten.sub.Tensor(where_37, mul_460);  where_37 = mul_460 = None
        sub_105: "f32[64, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_104, unsqueeze_345);  sub_104 = unsqueeze_345 = None
        mul_461: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_351);  sub_105 = unsqueeze_351 = None
        mul_462: "f32[240]" = torch.ops.aten.mul.Tensor(sum_56, squeeze_49);  sum_56 = squeeze_49 = None
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_461, div_8, primals_61, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_461 = div_8 = primals_61 = None
        getitem_155: "f32[64, 240, 14, 14]" = convolution_backward_29[0]
        getitem_156: "f32[240, 1, 5, 5]" = convolution_backward_29[1];  convolution_backward_29 = None
        lt_19: "b8[64, 240, 14, 14]" = torch.ops.aten.lt.Scalar(clone_107, -3)
        le_19: "b8[64, 240, 14, 14]" = torch.ops.aten.le.Scalar(clone_107, 3)
        div_48: "f32[64, 240, 14, 14]" = torch.ops.aten.div.Tensor(clone_107, 3);  clone_107 = None
        add_226: "f32[64, 240, 14, 14]" = torch.ops.aten.add.Tensor(div_48, 0.5);  div_48 = None
        mul_463: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_155, add_226);  add_226 = None
        where_38: "f32[64, 240, 14, 14]" = torch.ops.aten.where.self(le_19, mul_463, getitem_155);  le_19 = mul_463 = getitem_155 = None
        where_39: "f32[64, 240, 14, 14]" = torch.ops.aten.where.self(lt_19, scalar_tensor, where_38);  lt_19 = where_38 = None
        sum_57: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
        sub_106: "f32[64, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_354);  convolution_21 = unsqueeze_354 = None
        mul_464: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, sub_106)
        sum_58: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 2, 3]);  mul_464 = None
        mul_465: "f32[240]" = torch.ops.aten.mul.Tensor(sum_57, 7.971938775510203e-05)
        unsqueeze_355: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_465, 0);  mul_465 = None
        unsqueeze_356: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
        unsqueeze_357: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
        mul_466: "f32[240]" = torch.ops.aten.mul.Tensor(sum_58, 7.971938775510203e-05)
        mul_467: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_468: "f32[240]" = torch.ops.aten.mul.Tensor(mul_466, mul_467);  mul_466 = mul_467 = None
        unsqueeze_358: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
        unsqueeze_359: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
        unsqueeze_360: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
        mul_469: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_59);  primals_59 = None
        unsqueeze_361: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_469, 0);  mul_469 = None
        unsqueeze_362: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
        unsqueeze_363: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
        mul_470: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_360);  sub_106 = unsqueeze_360 = None
        sub_108: "f32[64, 240, 14, 14]" = torch.ops.aten.sub.Tensor(where_39, mul_470);  where_39 = mul_470 = None
        sub_109: "f32[64, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_108, unsqueeze_357);  sub_108 = unsqueeze_357 = None
        mul_471: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_363);  sub_109 = unsqueeze_363 = None
        mul_472: "f32[240]" = torch.ops.aten.mul.Tensor(sum_58, squeeze_46);  sum_58 = squeeze_46 = None
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_471, add_84, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_471 = add_84 = primals_58 = None
        getitem_158: "f32[64, 40, 14, 14]" = convolution_backward_30[0]
        getitem_159: "f32[240, 40, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        add_227: "f32[64, 40, 14, 14]" = torch.ops.aten.add.Tensor(getitem_143, getitem_158);  getitem_143 = getitem_158 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        sum_59: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_227, [0, 2, 3])
        sub_110: "f32[64, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_366);  convolution_20 = unsqueeze_366 = None
        mul_473: "f32[64, 40, 14, 14]" = torch.ops.aten.mul.Tensor(add_227, sub_110)
        sum_60: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_473, [0, 2, 3]);  mul_473 = None
        mul_474: "f32[40]" = torch.ops.aten.mul.Tensor(sum_59, 7.971938775510203e-05)
        unsqueeze_367: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_474, 0);  mul_474 = None
        unsqueeze_368: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
        unsqueeze_369: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
        mul_475: "f32[40]" = torch.ops.aten.mul.Tensor(sum_60, 7.971938775510203e-05)
        mul_476: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_477: "f32[40]" = torch.ops.aten.mul.Tensor(mul_475, mul_476);  mul_475 = mul_476 = None
        unsqueeze_370: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
        unsqueeze_371: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
        unsqueeze_372: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
        mul_478: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_56);  primals_56 = None
        unsqueeze_373: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_478, 0);  mul_478 = None
        unsqueeze_374: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
        unsqueeze_375: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
        mul_479: "f32[64, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_372);  sub_110 = unsqueeze_372 = None
        sub_112: "f32[64, 40, 14, 14]" = torch.ops.aten.sub.Tensor(add_227, mul_479);  mul_479 = None
        sub_113: "f32[64, 40, 14, 14]" = torch.ops.aten.sub.Tensor(sub_112, unsqueeze_369);  sub_112 = unsqueeze_369 = None
        mul_480: "f32[64, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_375);  sub_113 = unsqueeze_375 = None
        mul_481: "f32[40]" = torch.ops.aten.mul.Tensor(sum_60, squeeze_43);  sum_60 = squeeze_43 = None
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_480, mul_105, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_480 = mul_105 = primals_55 = None
        getitem_161: "f32[64, 240, 14, 14]" = convolution_backward_31[0]
        getitem_162: "f32[40, 240, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:260, code: return scale * input
        mul_482: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_161, div_7);  div_7 = None
        mul_483: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_161, div_6);  getitem_161 = div_6 = None
        sum_61: "f32[64, 240, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_483, [2, 3], True);  mul_483 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        mul_484: "f32[64, 240, 1, 1]" = torch.ops.aten.mul.Tensor(sum_61, 0.16666666666666666);  sum_61 = None
        where_40: "f32[64, 240, 1, 1]" = torch.ops.aten.where.self(bitwise_and_6, mul_484, scalar_tensor);  bitwise_and_6 = mul_484 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
        sum_62: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(where_40, relu_7, primals_53, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_40 = primals_53 = None
        getitem_164: "f32[64, 64, 1, 1]" = convolution_backward_32[0]
        getitem_165: "f32[240, 64, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
        le_20: "b8[64, 64, 1, 1]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_41: "f32[64, 64, 1, 1]" = torch.ops.aten.where.self(le_20, scalar_tensor, getitem_164);  le_20 = getitem_164 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
        sum_63: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(where_41, mean_2, primals_51, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_41 = mean_2 = primals_51 = None
        getitem_167: "f32[64, 240, 1, 1]" = convolution_backward_33[0]
        getitem_168: "f32[64, 240, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        expand_7: "f32[64, 240, 14, 14]" = torch.ops.aten.expand.default(getitem_167, [64, 240, 14, 14]);  getitem_167 = None
        div_49: "f32[64, 240, 14, 14]" = torch.ops.aten.div.Scalar(expand_7, 196);  expand_7 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        add_228: "f32[64, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_482, div_49);  mul_482 = div_49 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        lt_21: "b8[64, 240, 14, 14]" = torch.ops.aten.lt.Scalar(clone_106, -3)
        le_21: "b8[64, 240, 14, 14]" = torch.ops.aten.le.Scalar(clone_106, 3)
        div_50: "f32[64, 240, 14, 14]" = torch.ops.aten.div.Tensor(clone_106, 3);  clone_106 = None
        add_229: "f32[64, 240, 14, 14]" = torch.ops.aten.add.Tensor(div_50, 0.5);  div_50 = None
        mul_485: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(add_228, add_229);  add_229 = None
        where_42: "f32[64, 240, 14, 14]" = torch.ops.aten.where.self(le_21, mul_485, add_228);  le_21 = mul_485 = add_228 = None
        where_43: "f32[64, 240, 14, 14]" = torch.ops.aten.where.self(lt_21, scalar_tensor, where_42);  lt_21 = where_42 = None
        sum_64: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
        sub_114: "f32[64, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_378);  convolution_17 = unsqueeze_378 = None
        mul_486: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, sub_114)
        sum_65: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_486, [0, 2, 3]);  mul_486 = None
        mul_487: "f32[240]" = torch.ops.aten.mul.Tensor(sum_64, 7.971938775510203e-05)
        unsqueeze_379: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_487, 0);  mul_487 = None
        unsqueeze_380: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
        unsqueeze_381: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
        mul_488: "f32[240]" = torch.ops.aten.mul.Tensor(sum_65, 7.971938775510203e-05)
        mul_489: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_490: "f32[240]" = torch.ops.aten.mul.Tensor(mul_488, mul_489);  mul_488 = mul_489 = None
        unsqueeze_382: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_490, 0);  mul_490 = None
        unsqueeze_383: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
        unsqueeze_384: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
        mul_491: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_49);  primals_49 = None
        unsqueeze_385: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_491, 0);  mul_491 = None
        unsqueeze_386: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
        unsqueeze_387: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
        mul_492: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_384);  sub_114 = unsqueeze_384 = None
        sub_116: "f32[64, 240, 14, 14]" = torch.ops.aten.sub.Tensor(where_43, mul_492);  where_43 = mul_492 = None
        sub_117: "f32[64, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_116, unsqueeze_381);  sub_116 = unsqueeze_381 = None
        mul_493: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_387);  sub_117 = unsqueeze_387 = None
        mul_494: "f32[240]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_40);  sum_65 = squeeze_40 = None
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_493, div_5, primals_48, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_493 = div_5 = primals_48 = None
        getitem_170: "f32[64, 240, 14, 14]" = convolution_backward_34[0]
        getitem_171: "f32[240, 1, 5, 5]" = convolution_backward_34[1];  convolution_backward_34 = None
        lt_22: "b8[64, 240, 14, 14]" = torch.ops.aten.lt.Scalar(clone_105, -3)
        le_22: "b8[64, 240, 14, 14]" = torch.ops.aten.le.Scalar(clone_105, 3)
        div_51: "f32[64, 240, 14, 14]" = torch.ops.aten.div.Tensor(clone_105, 3);  clone_105 = None
        add_230: "f32[64, 240, 14, 14]" = torch.ops.aten.add.Tensor(div_51, 0.5);  div_51 = None
        mul_495: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_170, add_230);  add_230 = None
        where_44: "f32[64, 240, 14, 14]" = torch.ops.aten.where.self(le_22, mul_495, getitem_170);  le_22 = mul_495 = getitem_170 = None
        where_45: "f32[64, 240, 14, 14]" = torch.ops.aten.where.self(lt_22, scalar_tensor, where_44);  lt_22 = where_44 = None
        sum_66: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
        sub_118: "f32[64, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_390);  convolution_16 = unsqueeze_390 = None
        mul_496: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_45, sub_118)
        sum_67: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_496, [0, 2, 3]);  mul_496 = None
        mul_497: "f32[240]" = torch.ops.aten.mul.Tensor(sum_66, 7.971938775510203e-05)
        unsqueeze_391: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_497, 0);  mul_497 = None
        unsqueeze_392: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
        unsqueeze_393: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
        mul_498: "f32[240]" = torch.ops.aten.mul.Tensor(sum_67, 7.971938775510203e-05)
        mul_499: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_500: "f32[240]" = torch.ops.aten.mul.Tensor(mul_498, mul_499);  mul_498 = mul_499 = None
        unsqueeze_394: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_500, 0);  mul_500 = None
        unsqueeze_395: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
        unsqueeze_396: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
        mul_501: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_46);  primals_46 = None
        unsqueeze_397: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_501, 0);  mul_501 = None
        unsqueeze_398: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
        unsqueeze_399: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
        mul_502: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_396);  sub_118 = unsqueeze_396 = None
        sub_120: "f32[64, 240, 14, 14]" = torch.ops.aten.sub.Tensor(where_45, mul_502);  where_45 = mul_502 = None
        sub_121: "f32[64, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_120, unsqueeze_393);  sub_120 = unsqueeze_393 = None
        mul_503: "f32[64, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_399);  sub_121 = unsqueeze_399 = None
        mul_504: "f32[240]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_37);  sum_67 = squeeze_37 = None
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_503, add_65, primals_45, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_503 = add_65 = primals_45 = None
        getitem_173: "f32[64, 40, 14, 14]" = convolution_backward_35[0]
        getitem_174: "f32[240, 40, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        add_231: "f32[64, 40, 14, 14]" = torch.ops.aten.add.Tensor(add_227, getitem_173);  add_227 = getitem_173 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        sum_68: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_231, [0, 2, 3])
        sub_122: "f32[64, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_402);  convolution_15 = unsqueeze_402 = None
        mul_505: "f32[64, 40, 14, 14]" = torch.ops.aten.mul.Tensor(add_231, sub_122)
        sum_69: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_505, [0, 2, 3]);  mul_505 = None
        mul_506: "f32[40]" = torch.ops.aten.mul.Tensor(sum_68, 7.971938775510203e-05)
        unsqueeze_403: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_506, 0);  mul_506 = None
        unsqueeze_404: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
        unsqueeze_405: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
        mul_507: "f32[40]" = torch.ops.aten.mul.Tensor(sum_69, 7.971938775510203e-05)
        mul_508: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_509: "f32[40]" = torch.ops.aten.mul.Tensor(mul_507, mul_508);  mul_507 = mul_508 = None
        unsqueeze_406: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_509, 0);  mul_509 = None
        unsqueeze_407: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
        unsqueeze_408: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
        mul_510: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_43);  primals_43 = None
        unsqueeze_409: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_510, 0);  mul_510 = None
        unsqueeze_410: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
        unsqueeze_411: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
        mul_511: "f32[64, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_408);  sub_122 = unsqueeze_408 = None
        sub_124: "f32[64, 40, 14, 14]" = torch.ops.aten.sub.Tensor(add_231, mul_511);  add_231 = mul_511 = None
        sub_125: "f32[64, 40, 14, 14]" = torch.ops.aten.sub.Tensor(sub_124, unsqueeze_405);  sub_124 = unsqueeze_405 = None
        mul_512: "f32[64, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_411);  sub_125 = unsqueeze_411 = None
        mul_513: "f32[40]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_34);  sum_69 = squeeze_34 = None
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_512, mul_81, primals_42, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_512 = mul_81 = primals_42 = None
        getitem_176: "f32[64, 96, 14, 14]" = convolution_backward_36[0]
        getitem_177: "f32[40, 96, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:260, code: return scale * input
        mul_514: "f32[64, 96, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_176, div_4);  div_4 = None
        mul_515: "f32[64, 96, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_176, div_3);  getitem_176 = div_3 = None
        sum_70: "f32[64, 96, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_515, [2, 3], True);  mul_515 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        mul_516: "f32[64, 96, 1, 1]" = torch.ops.aten.mul.Tensor(sum_70, 0.16666666666666666);  sum_70 = None
        where_46: "f32[64, 96, 1, 1]" = torch.ops.aten.where.self(bitwise_and_7, mul_516, scalar_tensor);  bitwise_and_7 = mul_516 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
        sum_71: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(where_46, relu_6, primals_40, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_46 = primals_40 = None
        getitem_179: "f32[64, 24, 1, 1]" = convolution_backward_37[0]
        getitem_180: "f32[96, 24, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
        le_23: "b8[64, 24, 1, 1]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_47: "f32[64, 24, 1, 1]" = torch.ops.aten.where.self(le_23, scalar_tensor, getitem_179);  le_23 = getitem_179 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
        sum_72: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(where_47, mean_1, primals_38, [24], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_47 = mean_1 = primals_38 = None
        getitem_182: "f32[64, 96, 1, 1]" = convolution_backward_38[0]
        getitem_183: "f32[24, 96, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        expand_8: "f32[64, 96, 14, 14]" = torch.ops.aten.expand.default(getitem_182, [64, 96, 14, 14]);  getitem_182 = None
        div_52: "f32[64, 96, 14, 14]" = torch.ops.aten.div.Scalar(expand_8, 196);  expand_8 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        add_232: "f32[64, 96, 14, 14]" = torch.ops.aten.add.Tensor(mul_514, div_52);  mul_514 = div_52 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        lt_24: "b8[64, 96, 14, 14]" = torch.ops.aten.lt.Scalar(clone_104, -3)
        le_24: "b8[64, 96, 14, 14]" = torch.ops.aten.le.Scalar(clone_104, 3)
        div_53: "f32[64, 96, 14, 14]" = torch.ops.aten.div.Tensor(clone_104, 3);  clone_104 = None
        add_233: "f32[64, 96, 14, 14]" = torch.ops.aten.add.Tensor(div_53, 0.5);  div_53 = None
        mul_517: "f32[64, 96, 14, 14]" = torch.ops.aten.mul.Tensor(add_232, add_233);  add_233 = None
        where_48: "f32[64, 96, 14, 14]" = torch.ops.aten.where.self(le_24, mul_517, add_232);  le_24 = mul_517 = add_232 = None
        where_49: "f32[64, 96, 14, 14]" = torch.ops.aten.where.self(lt_24, scalar_tensor, where_48);  lt_24 = where_48 = None
        sum_73: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
        sub_126: "f32[64, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_414);  convolution_12 = unsqueeze_414 = None
        mul_518: "f32[64, 96, 14, 14]" = torch.ops.aten.mul.Tensor(where_49, sub_126)
        sum_74: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_518, [0, 2, 3]);  mul_518 = None
        mul_519: "f32[96]" = torch.ops.aten.mul.Tensor(sum_73, 7.971938775510203e-05)
        unsqueeze_415: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_519, 0);  mul_519 = None
        unsqueeze_416: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
        unsqueeze_417: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
        mul_520: "f32[96]" = torch.ops.aten.mul.Tensor(sum_74, 7.971938775510203e-05)
        mul_521: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_522: "f32[96]" = torch.ops.aten.mul.Tensor(mul_520, mul_521);  mul_520 = mul_521 = None
        unsqueeze_418: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_522, 0);  mul_522 = None
        unsqueeze_419: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
        unsqueeze_420: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
        mul_523: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_36);  primals_36 = None
        unsqueeze_421: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_523, 0);  mul_523 = None
        unsqueeze_422: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
        unsqueeze_423: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
        mul_524: "f32[64, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_420);  sub_126 = unsqueeze_420 = None
        sub_128: "f32[64, 96, 14, 14]" = torch.ops.aten.sub.Tensor(where_49, mul_524);  where_49 = mul_524 = None
        sub_129: "f32[64, 96, 14, 14]" = torch.ops.aten.sub.Tensor(sub_128, unsqueeze_417);  sub_128 = unsqueeze_417 = None
        mul_525: "f32[64, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_423);  sub_129 = unsqueeze_423 = None
        mul_526: "f32[96]" = torch.ops.aten.mul.Tensor(sum_74, squeeze_31);  sum_74 = squeeze_31 = None
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_525, div_2, primals_35, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 96, [True, True, False]);  mul_525 = div_2 = primals_35 = None
        getitem_185: "f32[64, 96, 28, 28]" = convolution_backward_39[0]
        getitem_186: "f32[96, 1, 5, 5]" = convolution_backward_39[1];  convolution_backward_39 = None
        lt_25: "b8[64, 96, 28, 28]" = torch.ops.aten.lt.Scalar(clone_103, -3)
        le_25: "b8[64, 96, 28, 28]" = torch.ops.aten.le.Scalar(clone_103, 3)
        div_54: "f32[64, 96, 28, 28]" = torch.ops.aten.div.Tensor(clone_103, 3);  clone_103 = None
        add_234: "f32[64, 96, 28, 28]" = torch.ops.aten.add.Tensor(div_54, 0.5);  div_54 = None
        mul_527: "f32[64, 96, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_185, add_234);  add_234 = None
        where_50: "f32[64, 96, 28, 28]" = torch.ops.aten.where.self(le_25, mul_527, getitem_185);  le_25 = mul_527 = getitem_185 = None
        where_51: "f32[64, 96, 28, 28]" = torch.ops.aten.where.self(lt_25, scalar_tensor, where_50);  lt_25 = where_50 = None
        sum_75: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
        sub_130: "f32[64, 96, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_426);  convolution_11 = unsqueeze_426 = None
        mul_528: "f32[64, 96, 28, 28]" = torch.ops.aten.mul.Tensor(where_51, sub_130)
        sum_76: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_528, [0, 2, 3]);  mul_528 = None
        mul_529: "f32[96]" = torch.ops.aten.mul.Tensor(sum_75, 1.992984693877551e-05)
        unsqueeze_427: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_529, 0);  mul_529 = None
        unsqueeze_428: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
        unsqueeze_429: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
        mul_530: "f32[96]" = torch.ops.aten.mul.Tensor(sum_76, 1.992984693877551e-05)
        mul_531: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_532: "f32[96]" = torch.ops.aten.mul.Tensor(mul_530, mul_531);  mul_530 = mul_531 = None
        unsqueeze_430: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_532, 0);  mul_532 = None
        unsqueeze_431: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
        unsqueeze_432: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
        mul_533: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_33);  primals_33 = None
        unsqueeze_433: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_533, 0);  mul_533 = None
        unsqueeze_434: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
        unsqueeze_435: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
        mul_534: "f32[64, 96, 28, 28]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_432);  sub_130 = unsqueeze_432 = None
        sub_132: "f32[64, 96, 28, 28]" = torch.ops.aten.sub.Tensor(where_51, mul_534);  where_51 = mul_534 = None
        sub_133: "f32[64, 96, 28, 28]" = torch.ops.aten.sub.Tensor(sub_132, unsqueeze_429);  sub_132 = unsqueeze_429 = None
        mul_535: "f32[64, 96, 28, 28]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_435);  sub_133 = unsqueeze_435 = None
        mul_536: "f32[96]" = torch.ops.aten.mul.Tensor(sum_76, squeeze_28);  sum_76 = squeeze_28 = None
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_535, add_47, primals_32, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_535 = add_47 = primals_32 = None
        getitem_188: "f32[64, 24, 28, 28]" = convolution_backward_40[0]
        getitem_189: "f32[96, 24, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
        sum_77: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_188, [0, 2, 3])
        sub_134: "f32[64, 24, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_438);  convolution_10 = unsqueeze_438 = None
        mul_537: "f32[64, 24, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_188, sub_134)
        sum_78: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_537, [0, 2, 3]);  mul_537 = None
        mul_538: "f32[24]" = torch.ops.aten.mul.Tensor(sum_77, 1.992984693877551e-05)
        unsqueeze_439: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_538, 0);  mul_538 = None
        unsqueeze_440: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
        unsqueeze_441: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
        mul_539: "f32[24]" = torch.ops.aten.mul.Tensor(sum_78, 1.992984693877551e-05)
        mul_540: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_541: "f32[24]" = torch.ops.aten.mul.Tensor(mul_539, mul_540);  mul_539 = mul_540 = None
        unsqueeze_442: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_541, 0);  mul_541 = None
        unsqueeze_443: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
        unsqueeze_444: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
        mul_542: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_30);  primals_30 = None
        unsqueeze_445: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_542, 0);  mul_542 = None
        unsqueeze_446: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
        unsqueeze_447: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
        mul_543: "f32[64, 24, 28, 28]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_444);  sub_134 = unsqueeze_444 = None
        sub_136: "f32[64, 24, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_188, mul_543);  mul_543 = None
        sub_137: "f32[64, 24, 28, 28]" = torch.ops.aten.sub.Tensor(sub_136, unsqueeze_441);  sub_136 = unsqueeze_441 = None
        mul_544: "f32[64, 24, 28, 28]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_447);  sub_137 = unsqueeze_447 = None
        mul_545: "f32[24]" = torch.ops.aten.mul.Tensor(sum_78, squeeze_25);  sum_78 = squeeze_25 = None
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_544, relu_5, primals_29, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_544 = primals_29 = None
        getitem_191: "f32[64, 88, 28, 28]" = convolution_backward_41[0]
        getitem_192: "f32[24, 88, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
        le_26: "b8[64, 88, 28, 28]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_52: "f32[64, 88, 28, 28]" = torch.ops.aten.where.self(le_26, scalar_tensor, getitem_191);  le_26 = getitem_191 = None
        sum_79: "f32[88]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
        sub_138: "f32[64, 88, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_450);  convolution_9 = unsqueeze_450 = None
        mul_546: "f32[64, 88, 28, 28]" = torch.ops.aten.mul.Tensor(where_52, sub_138)
        sum_80: "f32[88]" = torch.ops.aten.sum.dim_IntList(mul_546, [0, 2, 3]);  mul_546 = None
        mul_547: "f32[88]" = torch.ops.aten.mul.Tensor(sum_79, 1.992984693877551e-05)
        unsqueeze_451: "f32[1, 88]" = torch.ops.aten.unsqueeze.default(mul_547, 0);  mul_547 = None
        unsqueeze_452: "f32[1, 88, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
        unsqueeze_453: "f32[1, 88, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
        mul_548: "f32[88]" = torch.ops.aten.mul.Tensor(sum_80, 1.992984693877551e-05)
        mul_549: "f32[88]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_550: "f32[88]" = torch.ops.aten.mul.Tensor(mul_548, mul_549);  mul_548 = mul_549 = None
        unsqueeze_454: "f32[1, 88]" = torch.ops.aten.unsqueeze.default(mul_550, 0);  mul_550 = None
        unsqueeze_455: "f32[1, 88, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
        unsqueeze_456: "f32[1, 88, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
        mul_551: "f32[88]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_27);  primals_27 = None
        unsqueeze_457: "f32[1, 88]" = torch.ops.aten.unsqueeze.default(mul_551, 0);  mul_551 = None
        unsqueeze_458: "f32[1, 88, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
        unsqueeze_459: "f32[1, 88, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
        mul_552: "f32[64, 88, 28, 28]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_456);  sub_138 = unsqueeze_456 = None
        sub_140: "f32[64, 88, 28, 28]" = torch.ops.aten.sub.Tensor(where_52, mul_552);  where_52 = mul_552 = None
        sub_141: "f32[64, 88, 28, 28]" = torch.ops.aten.sub.Tensor(sub_140, unsqueeze_453);  sub_140 = unsqueeze_453 = None
        mul_553: "f32[64, 88, 28, 28]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_459);  sub_141 = unsqueeze_459 = None
        mul_554: "f32[88]" = torch.ops.aten.mul.Tensor(sum_80, squeeze_22);  sum_80 = squeeze_22 = None
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_553, relu_4, primals_26, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 88, [True, True, False]);  mul_553 = primals_26 = None
        getitem_194: "f32[64, 88, 28, 28]" = convolution_backward_42[0]
        getitem_195: "f32[88, 1, 3, 3]" = convolution_backward_42[1];  convolution_backward_42 = None
        le_27: "b8[64, 88, 28, 28]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_53: "f32[64, 88, 28, 28]" = torch.ops.aten.where.self(le_27, scalar_tensor, getitem_194);  le_27 = getitem_194 = None
        sum_81: "f32[88]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
        sub_142: "f32[64, 88, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_462);  convolution_8 = unsqueeze_462 = None
        mul_555: "f32[64, 88, 28, 28]" = torch.ops.aten.mul.Tensor(where_53, sub_142)
        sum_82: "f32[88]" = torch.ops.aten.sum.dim_IntList(mul_555, [0, 2, 3]);  mul_555 = None
        mul_556: "f32[88]" = torch.ops.aten.mul.Tensor(sum_81, 1.992984693877551e-05)
        unsqueeze_463: "f32[1, 88]" = torch.ops.aten.unsqueeze.default(mul_556, 0);  mul_556 = None
        unsqueeze_464: "f32[1, 88, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
        unsqueeze_465: "f32[1, 88, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
        mul_557: "f32[88]" = torch.ops.aten.mul.Tensor(sum_82, 1.992984693877551e-05)
        mul_558: "f32[88]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_559: "f32[88]" = torch.ops.aten.mul.Tensor(mul_557, mul_558);  mul_557 = mul_558 = None
        unsqueeze_466: "f32[1, 88]" = torch.ops.aten.unsqueeze.default(mul_559, 0);  mul_559 = None
        unsqueeze_467: "f32[1, 88, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
        unsqueeze_468: "f32[1, 88, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
        mul_560: "f32[88]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_24);  primals_24 = None
        unsqueeze_469: "f32[1, 88]" = torch.ops.aten.unsqueeze.default(mul_560, 0);  mul_560 = None
        unsqueeze_470: "f32[1, 88, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
        unsqueeze_471: "f32[1, 88, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
        mul_561: "f32[64, 88, 28, 28]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_468);  sub_142 = unsqueeze_468 = None
        sub_144: "f32[64, 88, 28, 28]" = torch.ops.aten.sub.Tensor(where_53, mul_561);  where_53 = mul_561 = None
        sub_145: "f32[64, 88, 28, 28]" = torch.ops.aten.sub.Tensor(sub_144, unsqueeze_465);  sub_144 = unsqueeze_465 = None
        mul_562: "f32[64, 88, 28, 28]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_471);  sub_145 = unsqueeze_471 = None
        mul_563: "f32[88]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_19);  sum_82 = squeeze_19 = None
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_562, add_31, primals_23, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_562 = add_31 = primals_23 = None
        getitem_197: "f32[64, 24, 28, 28]" = convolution_backward_43[0]
        getitem_198: "f32[88, 24, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        add_235: "f32[64, 24, 28, 28]" = torch.ops.aten.add.Tensor(getitem_188, getitem_197);  getitem_188 = getitem_197 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        sum_83: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_235, [0, 2, 3])
        sub_146: "f32[64, 24, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_474);  convolution_7 = unsqueeze_474 = None
        mul_564: "f32[64, 24, 28, 28]" = torch.ops.aten.mul.Tensor(add_235, sub_146)
        sum_84: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_564, [0, 2, 3]);  mul_564 = None
        mul_565: "f32[24]" = torch.ops.aten.mul.Tensor(sum_83, 1.992984693877551e-05)
        unsqueeze_475: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_565, 0);  mul_565 = None
        unsqueeze_476: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
        unsqueeze_477: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
        mul_566: "f32[24]" = torch.ops.aten.mul.Tensor(sum_84, 1.992984693877551e-05)
        mul_567: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_568: "f32[24]" = torch.ops.aten.mul.Tensor(mul_566, mul_567);  mul_566 = mul_567 = None
        unsqueeze_478: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_568, 0);  mul_568 = None
        unsqueeze_479: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
        unsqueeze_480: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
        mul_569: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_21);  primals_21 = None
        unsqueeze_481: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_569, 0);  mul_569 = None
        unsqueeze_482: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
        unsqueeze_483: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
        mul_570: "f32[64, 24, 28, 28]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_480);  sub_146 = unsqueeze_480 = None
        sub_148: "f32[64, 24, 28, 28]" = torch.ops.aten.sub.Tensor(add_235, mul_570);  add_235 = mul_570 = None
        sub_149: "f32[64, 24, 28, 28]" = torch.ops.aten.sub.Tensor(sub_148, unsqueeze_477);  sub_148 = unsqueeze_477 = None
        mul_571: "f32[64, 24, 28, 28]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_483);  sub_149 = unsqueeze_483 = None
        mul_572: "f32[24]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_16);  sum_84 = squeeze_16 = None
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_571, relu_3, primals_20, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_571 = primals_20 = None
        getitem_200: "f32[64, 72, 28, 28]" = convolution_backward_44[0]
        getitem_201: "f32[24, 72, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
        le_28: "b8[64, 72, 28, 28]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_54: "f32[64, 72, 28, 28]" = torch.ops.aten.where.self(le_28, scalar_tensor, getitem_200);  le_28 = getitem_200 = None
        sum_85: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
        sub_150: "f32[64, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_486);  convolution_6 = unsqueeze_486 = None
        mul_573: "f32[64, 72, 28, 28]" = torch.ops.aten.mul.Tensor(where_54, sub_150)
        sum_86: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_573, [0, 2, 3]);  mul_573 = None
        mul_574: "f32[72]" = torch.ops.aten.mul.Tensor(sum_85, 1.992984693877551e-05)
        unsqueeze_487: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_574, 0);  mul_574 = None
        unsqueeze_488: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
        unsqueeze_489: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
        mul_575: "f32[72]" = torch.ops.aten.mul.Tensor(sum_86, 1.992984693877551e-05)
        mul_576: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_577: "f32[72]" = torch.ops.aten.mul.Tensor(mul_575, mul_576);  mul_575 = mul_576 = None
        unsqueeze_490: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_577, 0);  mul_577 = None
        unsqueeze_491: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
        unsqueeze_492: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
        mul_578: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_18);  primals_18 = None
        unsqueeze_493: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_578, 0);  mul_578 = None
        unsqueeze_494: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
        unsqueeze_495: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
        mul_579: "f32[64, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_492);  sub_150 = unsqueeze_492 = None
        sub_152: "f32[64, 72, 28, 28]" = torch.ops.aten.sub.Tensor(where_54, mul_579);  where_54 = mul_579 = None
        sub_153: "f32[64, 72, 28, 28]" = torch.ops.aten.sub.Tensor(sub_152, unsqueeze_489);  sub_152 = unsqueeze_489 = None
        mul_580: "f32[64, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_495);  sub_153 = unsqueeze_495 = None
        mul_581: "f32[72]" = torch.ops.aten.mul.Tensor(sum_86, squeeze_13);  sum_86 = squeeze_13 = None
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_580, relu_2, primals_17, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_580 = primals_17 = None
        getitem_203: "f32[64, 72, 56, 56]" = convolution_backward_45[0]
        getitem_204: "f32[72, 1, 3, 3]" = convolution_backward_45[1];  convolution_backward_45 = None
        le_29: "b8[64, 72, 56, 56]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_55: "f32[64, 72, 56, 56]" = torch.ops.aten.where.self(le_29, scalar_tensor, getitem_203);  le_29 = getitem_203 = None
        sum_87: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
        sub_154: "f32[64, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_498);  convolution_5 = unsqueeze_498 = None
        mul_582: "f32[64, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_55, sub_154)
        sum_88: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_582, [0, 2, 3]);  mul_582 = None
        mul_583: "f32[72]" = torch.ops.aten.mul.Tensor(sum_87, 4.982461734693877e-06)
        unsqueeze_499: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_583, 0);  mul_583 = None
        unsqueeze_500: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
        unsqueeze_501: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
        mul_584: "f32[72]" = torch.ops.aten.mul.Tensor(sum_88, 4.982461734693877e-06)
        mul_585: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_586: "f32[72]" = torch.ops.aten.mul.Tensor(mul_584, mul_585);  mul_584 = mul_585 = None
        unsqueeze_502: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_586, 0);  mul_586 = None
        unsqueeze_503: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
        unsqueeze_504: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
        mul_587: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_15);  primals_15 = None
        unsqueeze_505: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_587, 0);  mul_587 = None
        unsqueeze_506: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
        unsqueeze_507: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
        mul_588: "f32[64, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_504);  sub_154 = unsqueeze_504 = None
        sub_156: "f32[64, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_55, mul_588);  where_55 = mul_588 = None
        sub_157: "f32[64, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_156, unsqueeze_501);  sub_156 = unsqueeze_501 = None
        mul_589: "f32[64, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_507);  sub_157 = unsqueeze_507 = None
        mul_590: "f32[72]" = torch.ops.aten.mul.Tensor(sum_88, squeeze_10);  sum_88 = squeeze_10 = None
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_589, add_16, primals_14, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_589 = add_16 = primals_14 = None
        getitem_206: "f32[64, 16, 56, 56]" = convolution_backward_46[0]
        getitem_207: "f32[72, 16, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
        sum_89: "f32[16]" = torch.ops.aten.sum.dim_IntList(getitem_206, [0, 2, 3])
        sub_158: "f32[64, 16, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_510);  convolution_4 = unsqueeze_510 = None
        mul_591: "f32[64, 16, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_206, sub_158)
        sum_90: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_591, [0, 2, 3]);  mul_591 = None
        mul_592: "f32[16]" = torch.ops.aten.mul.Tensor(sum_89, 4.982461734693877e-06)
        unsqueeze_511: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_592, 0);  mul_592 = None
        unsqueeze_512: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
        unsqueeze_513: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
        mul_593: "f32[16]" = torch.ops.aten.mul.Tensor(sum_90, 4.982461734693877e-06)
        mul_594: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_595: "f32[16]" = torch.ops.aten.mul.Tensor(mul_593, mul_594);  mul_593 = mul_594 = None
        unsqueeze_514: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_595, 0);  mul_595 = None
        unsqueeze_515: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
        unsqueeze_516: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
        mul_596: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_12);  primals_12 = None
        unsqueeze_517: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_596, 0);  mul_596 = None
        unsqueeze_518: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
        unsqueeze_519: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
        mul_597: "f32[64, 16, 56, 56]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_516);  sub_158 = unsqueeze_516 = None
        sub_160: "f32[64, 16, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_206, mul_597);  getitem_206 = mul_597 = None
        sub_161: "f32[64, 16, 56, 56]" = torch.ops.aten.sub.Tensor(sub_160, unsqueeze_513);  sub_160 = unsqueeze_513 = None
        mul_598: "f32[64, 16, 56, 56]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_519);  sub_161 = unsqueeze_519 = None
        mul_599: "f32[16]" = torch.ops.aten.mul.Tensor(sum_90, squeeze_7);  sum_90 = squeeze_7 = None
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_598, mul_15, primals_11, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_598 = mul_15 = primals_11 = None
        getitem_209: "f32[64, 16, 56, 56]" = convolution_backward_47[0]
        getitem_210: "f32[16, 16, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:260, code: return scale * input
        mul_600: "f32[64, 16, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_209, div_1);  div_1 = None
        mul_601: "f32[64, 16, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_209, relu);  getitem_209 = None
        sum_91: "f32[64, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_601, [2, 3], True);  mul_601 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
        mul_602: "f32[64, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sum_91, 0.16666666666666666);  sum_91 = None
        where_56: "f32[64, 16, 1, 1]" = torch.ops.aten.where.self(bitwise_and_8, mul_602, scalar_tensor);  bitwise_and_8 = mul_602 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
        sum_92: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(where_56, relu_1, primals_9, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_56 = primals_9 = None
        getitem_212: "f32[64, 8, 1, 1]" = convolution_backward_48[0]
        getitem_213: "f32[16, 8, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
        le_30: "b8[64, 8, 1, 1]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_57: "f32[64, 8, 1, 1]" = torch.ops.aten.where.self(le_30, scalar_tensor, getitem_212);  le_30 = getitem_212 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
        sum_93: "f32[8]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(where_57, mean, primals_7, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_57 = mean = primals_7 = None
        getitem_215: "f32[64, 16, 1, 1]" = convolution_backward_49[0]
        getitem_216: "f32[8, 16, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        expand_9: "f32[64, 16, 56, 56]" = torch.ops.aten.expand.default(getitem_215, [64, 16, 56, 56]);  getitem_215 = None
        div_55: "f32[64, 16, 56, 56]" = torch.ops.aten.div.Scalar(expand_9, 3136);  expand_9 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
        add_236: "f32[64, 16, 56, 56]" = torch.ops.aten.add.Tensor(mul_600, div_55);  mul_600 = div_55 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
        le_31: "b8[64, 16, 56, 56]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_58: "f32[64, 16, 56, 56]" = torch.ops.aten.where.self(le_31, scalar_tensor, add_236);  le_31 = add_236 = None
        sum_94: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
        sub_162: "f32[64, 16, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_522);  convolution_1 = unsqueeze_522 = None
        mul_603: "f32[64, 16, 56, 56]" = torch.ops.aten.mul.Tensor(where_58, sub_162)
        sum_95: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_603, [0, 2, 3]);  mul_603 = None
        mul_604: "f32[16]" = torch.ops.aten.mul.Tensor(sum_94, 4.982461734693877e-06)
        unsqueeze_523: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_604, 0);  mul_604 = None
        unsqueeze_524: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
        unsqueeze_525: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
        mul_605: "f32[16]" = torch.ops.aten.mul.Tensor(sum_95, 4.982461734693877e-06)
        mul_606: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_607: "f32[16]" = torch.ops.aten.mul.Tensor(mul_605, mul_606);  mul_605 = mul_606 = None
        unsqueeze_526: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_607, 0);  mul_607 = None
        unsqueeze_527: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
        unsqueeze_528: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
        mul_608: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_5);  primals_5 = None
        unsqueeze_529: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_608, 0);  mul_608 = None
        unsqueeze_530: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
        unsqueeze_531: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
        mul_609: "f32[64, 16, 56, 56]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_528);  sub_162 = unsqueeze_528 = None
        sub_164: "f32[64, 16, 56, 56]" = torch.ops.aten.sub.Tensor(where_58, mul_609);  where_58 = mul_609 = None
        sub_165: "f32[64, 16, 56, 56]" = torch.ops.aten.sub.Tensor(sub_164, unsqueeze_525);  sub_164 = unsqueeze_525 = None
        mul_610: "f32[64, 16, 56, 56]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_531);  sub_165 = unsqueeze_531 = None
        mul_611: "f32[16]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_4);  sum_95 = squeeze_4 = None
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_610, div, primals_4, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False]);  mul_610 = div = primals_4 = None
        getitem_218: "f32[64, 16, 112, 112]" = convolution_backward_50[0]
        getitem_219: "f32[16, 1, 3, 3]" = convolution_backward_50[1];  convolution_backward_50 = None
        
        # File: /home/zibo/.local/lib/python3.8/site-packages/torchvision/models/mobilenetv3.py:210, code: x = self.features(x)
        lt_27: "b8[64, 16, 112, 112]" = torch.ops.aten.lt.Scalar(clone_102, -3)
        le_32: "b8[64, 16, 112, 112]" = torch.ops.aten.le.Scalar(clone_102, 3)
        div_56: "f32[64, 16, 112, 112]" = torch.ops.aten.div.Tensor(clone_102, 3);  clone_102 = None
        add_237: "f32[64, 16, 112, 112]" = torch.ops.aten.add.Tensor(div_56, 0.5);  div_56 = None
        mul_612: "f32[64, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_218, add_237);  add_237 = None
        where_59: "f32[64, 16, 112, 112]" = torch.ops.aten.where.self(le_32, mul_612, getitem_218);  le_32 = mul_612 = getitem_218 = None
        where_60: "f32[64, 16, 112, 112]" = torch.ops.aten.where.self(lt_27, scalar_tensor, where_59);  lt_27 = scalar_tensor = where_59 = None
        sum_96: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
        sub_166: "f32[64, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_534);  convolution = unsqueeze_534 = None
        mul_613: "f32[64, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_60, sub_166)
        sum_97: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_613, [0, 2, 3]);  mul_613 = None
        mul_614: "f32[16]" = torch.ops.aten.mul.Tensor(sum_96, 1.2456154336734693e-06)
        unsqueeze_535: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_614, 0);  mul_614 = None
        unsqueeze_536: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
        unsqueeze_537: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
        mul_615: "f32[16]" = torch.ops.aten.mul.Tensor(sum_97, 1.2456154336734693e-06)
        mul_616: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_617: "f32[16]" = torch.ops.aten.mul.Tensor(mul_615, mul_616);  mul_615 = mul_616 = None
        unsqueeze_538: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_617, 0);  mul_617 = None
        unsqueeze_539: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
        unsqueeze_540: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
        mul_618: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_2);  primals_2 = None
        unsqueeze_541: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_618, 0);  mul_618 = None
        unsqueeze_542: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
        unsqueeze_543: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
        mul_619: "f32[64, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_540);  sub_166 = unsqueeze_540 = None
        sub_168: "f32[64, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_60, mul_619);  where_60 = mul_619 = None
        sub_169: "f32[64, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_168, unsqueeze_537);  sub_168 = unsqueeze_537 = None
        mul_620: "f32[64, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_543);  sub_169 = unsqueeze_543 = None
        mul_621: "f32[16]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_1);  sum_97 = squeeze_1 = None
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_620, primals_245, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_620 = primals_245 = primals_1 = None
        getitem_221: "f32[64, 3, 224, 224]" = convolution_backward_51[0]
        getitem_222: "f32[16, 3, 3, 3]" = convolution_backward_51[1];  convolution_backward_51 = None
        return [getitem_222, mul_621, sum_96, getitem_219, mul_611, sum_94, getitem_216, sum_93, getitem_213, sum_92, getitem_210, mul_599, sum_89, getitem_207, mul_590, sum_87, getitem_204, mul_581, sum_85, getitem_201, mul_572, sum_83, getitem_198, mul_563, sum_81, getitem_195, mul_554, sum_79, getitem_192, mul_545, sum_77, getitem_189, mul_536, sum_75, getitem_186, mul_526, sum_73, getitem_183, sum_72, getitem_180, sum_71, getitem_177, mul_513, sum_68, getitem_174, mul_504, sum_66, getitem_171, mul_494, sum_64, getitem_168, sum_63, getitem_165, sum_62, getitem_162, mul_481, sum_59, getitem_159, mul_472, sum_57, getitem_156, mul_462, sum_55, getitem_153, sum_54, getitem_150, sum_53, getitem_147, mul_449, sum_50, getitem_144, mul_440, sum_48, getitem_141, mul_430, sum_46, getitem_138, sum_45, getitem_135, sum_44, getitem_132, mul_417, sum_41, getitem_129, mul_408, sum_39, getitem_126, mul_398, sum_37, getitem_123, sum_36, getitem_120, sum_35, getitem_117, mul_385, sum_32, getitem_114, mul_376, sum_30, getitem_111, mul_366, sum_28, getitem_108, sum_27, getitem_105, sum_26, getitem_102, mul_353, sum_23, getitem_99, mul_344, sum_21, getitem_96, mul_334, sum_19, getitem_93, sum_18, getitem_90, sum_17, getitem_87, mul_321, sum_14, getitem_84, mul_312, sum_12, getitem_81, mul_302, sum_10, getitem_78, sum_9, getitem_75, sum_8, getitem_72, mul_289, sum_5, getitem_69, mul_280, sum_3, permute_9, view_2, permute_5, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, getitem_221]
        
