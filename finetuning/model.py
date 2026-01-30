import torch

class CustomModel(torch.nn.Module):
    def __init__(self, base_model, args):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        self.args = args
        if args.model == "resnet50":
            if args.intermediate_layer:
                intermediate_dim = args.model_dim * args.ex_factor
                if args.act_type == 'topk':
                    self.intermediate_layer = torch.nn.Linear(args.model_dim, intermediate_dim)
                else:
                    raise NotImplementedError(f"Activation type {args.act_type} not implemented")
                self.classifier = torch.nn.Linear(intermediate_dim, args.num_classes)
            else:
                if 'imagenet' in args.data_path.lower():
                    self.classifier = self.base_model.fc # move the original head to a new attribute
                else:
                    self.classifier = torch.nn.Linear(2048, args.num_classes) # new classification head
            self.base_model.fc = torch.nn.Identity() # replace the original head with identity

        elif args.model == 'clip_vit_b_32': # new classification head
            if args.intermediate_layer:
                intermediate_dim = args.model_dim * args.ex_factor
                if args.act_type == 'topk':
                    self.intermediate_layer = torch.nn.Linear(args.model_dim, intermediate_dim)
                else:
                    raise NotImplementedError(f"Activation type {args.act_type} not implemented")
                self.classifier = torch.nn.Linear(intermediate_dim, args.num_classes)
            else:
                self.classifier = torch.nn.Linear(args.model_dim, args.num_classes)


    def forward(self, x):
        if self.args.model == 'clip_vit_b_32':
            output_before_head = self.base_model(pixel_values=x).pooler_output # CLS token
            if self.args.intermediate_layer:
                output_before_head = self.intermediate_layer(output_before_head)
                if self.args.act_type == 'topk':
                    output_before_head = torch.relu(output_before_head) # apply ReLU before TopK
                    post_topk = output_before_head.topk(self.args.k, sorted=False, dim=-1) # get TopK activations
                    tops_acts_BK = post_topk.values
                    top_indices_BK = post_topk.indices
                    buffer_BF = torch.zeros_like(output_before_head)
                    output_before_head = buffer_BF.scatter_(dim=-1, index=top_indices_BK, src=tops_acts_BK)
            
            output_after_head = self.classifier(output_before_head)
            return output_before_head, output_after_head 
        else:
            output_before_head = self.base_model(x)
            if self.args.intermediate_layer:
                output_before_head = self.intermediate_layer(output_before_head)
                if self.args.act_type == 'topk':
                    output_before_head = torch.relu(output_before_head) # apply ReLU before TopK
                    post_topk = output_before_head.topk(self.args.k, sorted=False, dim=-1) # get TopK activations
                    tops_acts_BK = post_topk.values
                    top_indices_BK = post_topk.indices
                    buffer_BF = torch.zeros_like(output_before_head)
                    output_before_head = buffer_BF.scatter_(dim=-1, index=top_indices_BK, src=tops_acts_BK)
            output_after_head = self.classifier(output_before_head)
        return output_before_head, output_after_head