import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, init_weights=False):  #注意参数weights
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(      #当网络层数较多时使用Sequential函数一起打包为features
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ELU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ELU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(        #同上将全连接层一起打包为classifier
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ELU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:  #初始化权重
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)  #网络模块
        x = torch.flatten(x, start_dim=1)  #展平从第二个维度开始
        x = self.classifier(x)  #分类模块
        return x

    def _initialize_weights(self):  #权重定义
        for m in self.modules():    #遍历网络层
            if isinstance(m, nn.Conv2d):  #如果为卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu',a=1)  #kaiming初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  #全连接层
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
