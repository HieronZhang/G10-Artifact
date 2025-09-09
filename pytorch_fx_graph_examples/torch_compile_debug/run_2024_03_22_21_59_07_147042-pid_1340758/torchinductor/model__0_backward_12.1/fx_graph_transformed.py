class GraphModule(torch.nn.Module):
    def forward(self, primals_7: "f32[64, 784]", relu: "f32[64, 512]", relu_1: "f32[64, 512]", permute_3: "f32[10, 512]", permute_7: "f32[512, 512]", tangents_1: "f32[64, 10]"):
        # File: /tmp/ipykernel_1340758/3322860944.py:24, code: logits = self.linear_relu_stack(x)
        mm: "f32[64, 512]" = torch.ops.aten.mm.default(tangents_1, permute_3);  permute_3 = None
        permute_4: "f32[10, 64]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "f32[10, 512]" = torch.ops.aten.mm.default(permute_4, relu_1);  permute_4 = None
        permute_5: "f32[512, 10]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        sum_1: "f32[1, 10]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view: "f32[10]" = torch.ops.aten.reshape.default(sum_1, [10]);  sum_1 = None
        permute_6: "f32[10, 512]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        le: "b8[64, 512]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[64, 512]" = torch.ops.aten.where.self(le, full_default, mm);  le = mm = None
        mm_2: "f32[64, 512]" = torch.ops.aten.mm.default(where, permute_7);  permute_7 = None
        permute_8: "f32[512, 64]" = torch.ops.aten.permute.default(where, [1, 0])
        mm_3: "f32[512, 512]" = torch.ops.aten.mm.default(permute_8, relu);  permute_8 = None
        permute_9: "f32[512, 512]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
        sum_2: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(where, [0], True);  where = None
        view_1: "f32[512]" = torch.ops.aten.reshape.default(sum_2, [512]);  sum_2 = None
        permute_10: "f32[512, 512]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        le_1: "b8[64, 512]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_1: "f32[64, 512]" = torch.ops.aten.where.self(le_1, full_default, mm_2);  le_1 = full_default = mm_2 = None
        permute_11: "f32[512, 64]" = torch.ops.aten.permute.default(where_1, [1, 0])
        mm_4: "f32[512, 784]" = torch.ops.aten.mm.default(permute_11, primals_7);  permute_11 = primals_7 = None
        permute_12: "f32[784, 512]" = torch.ops.aten.permute.default(mm_4, [1, 0]);  mm_4 = None
        sum_3: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(where_1, [0], True);  where_1 = None
        view_2: "f32[512]" = torch.ops.aten.reshape.default(sum_3, [512]);  sum_3 = None
        permute_13: "f32[512, 784]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        return [permute_13, view_2, permute_10, view_1, permute_6, view, None]
        