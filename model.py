import torch.nn as nn
import torchvision

pretrained_resnet50_model = torchvision.models.resnet50(pretrained=True)


class BB_model(nn.Module):
    def __init__(self):
        super(BB_model, self).__init__()
        self.defaultLayers = nn.Sequential(*list(pretrained_resnet50_model.children())[:-1])
        self.classifier = nn.Sequential(nn.Linear(2048, 3))
        self.bb = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, input):
        input = self.defaultLayers(input)
        input = input.view(input.shape[0], -1)
        return self.classifier(input), self.bb(input)
