class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[512, 784]", primals_2: "f32[512]", primals_3: "f32[512, 512]", primals_4: "f32[512]", primals_5: "f32[10, 512]", primals_6: "f32[10]", primals_7: "f32[64, 784]"):
        # File: /tmp/ipykernel_1340758/3322860944.py:24, code: logits = self.linear_relu_stack(x)
        permute: "f32[784, 512]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        addmm: "f32[64, 512]" = torch.ops.aten.addmm.default(primals_2, primals_7, permute);  primals_2 = permute = None
        relu: "f32[64, 512]" = torch.ops.aten.relu.default(addmm);  addmm = None
        permute_1: "f32[512, 512]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        addmm_1: "f32[64, 512]" = torch.ops.aten.addmm.default(primals_4, relu, permute_1);  primals_4 = None
        relu_1: "f32[64, 512]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        permute_2: "f32[512, 10]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        addmm_2: "f32[64, 10]" = torch.ops.aten.addmm.default(primals_6, relu_1, permute_2);  primals_6 = None
        permute_3: "f32[10, 512]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_7: "f32[512, 512]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return [addmm_2, primals_7, relu, relu_1, permute_3, permute_7]
        