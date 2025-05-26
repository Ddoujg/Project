import torch.nn as nn
import torch

class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        #self.avgpool = nn.AdaptiveAvgPool1d(1)  # 自适应平均池化
        self.classifier = nn.Sequential(
            nn.Linear(512*7, 4096),  # 修改输入维度为512
            nn.ELU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ELU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)  # [batch, 512, 1]
        x = torch.flatten(x, 1)  # [batch, 512]
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):  # 初始化Conv1d
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

def make_features(cfg: list, in_channels=3):  # 添加输入通道参数
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]  # 一维池化
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)  # 一维卷积
            layers += [conv1d, nn.ELU(True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg1d(model_name="vgg16",  **kwargs):
    assert model_name in cfgs, f"Model {model_name} not supported"
    cfg = cfgs[model_name]
    model = VGG(make_features(cfg), **kwargs)
    return model