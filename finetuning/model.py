import torchvision
import torch

class CustomModel(torch.nn.Module):
    def __init__(self, base_model, args):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        if args.model == "resnet50":
            self.classifier = self.base_model.fc     # move the original head to a new attribute
            self.base_model.fc = torch.nn.Identity() # replace the original head with identity
        elif args.model.startswith('vit_'):
            self.classifier = self.base_model.heads.head
            self.base_model.heads.head = torch.nn.Identity()

    def forward(self, x):
        output_before_head = self.base_model(x)
        output_after_head = self.classifier(output_before_head)
        return output_before_head, output_after_head