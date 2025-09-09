class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[512, 784]", primals_2: "f32[512]", primals_3: "f32[512, 512]", primals_4: "f32[512]", primals_5: "f32[10, 512]", primals_6: "f32[10]", primals_7: "f32[64, 784]"):
        # File: /tmp/ipykernel_1340758/3322860944.py:24, code: logits = self.linear_relu_stack(x)
        permute: "f32[784, 512]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        
        # No stacktrace found for following nodes
        mm_default_1: "f32[64, 512]" = torch.ops.aten.mm.default(primals_7, permute);  permute = None
        add_tensor_1: "f32[64, 512]" = torch.ops.aten.add.Tensor(mm_default_1, primals_2);  mm_default_1 = primals_2 = None
        
        # File: /tmp/ipykernel_1340758/3322860944.py:24, code: logits = self.linear_relu_stack(x)
        relu: "f32[64, 512]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        permute_1: "f32[512, 512]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        
        # No stacktrace found for following nodes
        mm_default: "f32[64, 512]" = torch.ops.aten.mm.default(relu, permute_1)
        add_tensor: "f32[64, 512]" = torch.ops.aten.add.Tensor(mm_default, primals_4);  mm_default = primals_4 = None
        
        # File: /tmp/ipykernel_1340758/3322860944.py:24, code: logits = self.linear_relu_stack(x)
        relu_1: "f32[64, 512]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        permute_2: "f32[512, 10]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        addmm_2: "f32[64, 10]" = torch.ops.aten.addmm.default(primals_6, relu_1, permute_2);  primals_6 = None
        permute_3: "f32[10, 512]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_7: "f32[512, 512]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return [addmm_2, primals_7, relu, relu_1, permute_3, permute_7]
        