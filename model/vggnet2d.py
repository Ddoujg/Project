import torch.nn as nn
import torch

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ELU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ELU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:  #是否进行权重初始化
            self._initialize_weights()

    def forward(self, x):
        # N x 1 x 224 x 224
        x = self.features(x)
        # N x 224 x 1 x 1
        x = torch.flatten(x, start_dim=1)
        # N x 512*1*1
        x = self.classifier(x)
        return x

    # def _initialize_weights(self):  #初始化权重定义
    #     for m in self.modules():  #遍历网络层
    #         if isinstance(m, nn.Conv2d):  #如果为卷积层
    #             # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):  #如果为全连接层
    #             nn.init.xavier_uniform_(m.weight)
    #             # nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)


def make_features(cfg: list):  #传入配置文件
    layers = []
    in_channels = 1
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ELU(True)]
            in_channels = v
    return nn.Sequential(*layers)  #通过非关键字参数传入列表


cfgs = {  #四种配置文件 字典
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg2d(model_name="vgg16", **kwargs):  #**kwargs为字典变量包含num_classes、init_weights等变量
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model
