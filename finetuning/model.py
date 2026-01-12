import torch

class CustomModel(torch.nn.Module):
    def __init__(self, base_model, args):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        self.args = args
        if args.model == "resnet50":
            if 'imagenet' in args.data_path.lower():
                self.classifier = self.base_model.fc     # move the original head to a new attribute
            elif 'cifar10' in args.data_path.lower():
                self.classifier = torch.nn.Linear(2048, 10)  
            elif 'cifar100' in args.data_path.lower():
                self.classifier = torch.nn.Linear(2048, 100)
            self.base_model.fc = torch.nn.Identity() # replace the original head with identity

        elif args.model == 'clip_vit_b_32': # new classification head
            if 'imagenet' in args.data_path.lower():
                self.classifier = torch.nn.Linear(768, 1000)  
            elif 'cifar10' in args.data_path.lower():
                self.classifier = torch.nn.Linear(768, 10)
            elif 'cifar100' in args.data_path.lower():
                self.classifier = torch.nn.Linear(768, 100)

    def forward(self, x):
        if self.args.model == 'clip_vit_b_32':
            output_before_head = self.base_model(pixel_values=x).pooler_output # CLS token
            output_after_head = self.classifier(output_before_head)
            return output_before_head, output_after_head 
        else:
            output_before_head = self.base_model(x)
            output_after_head = self.classifier(output_before_head)
        return output_before_head, output_after_head