from torch import nn
from MVDFE_Module import MVDFE_Module
from classifier import Class_Specific_Classifier



class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, attention=False, attention_kernel_size=7):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.attention = attention

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if self.attention == True:
            self.mvdfe = MVDFE_Module(kernel_size=attention_kernel_size)

        if stride == 2:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        if self.stride == 2:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        if self.attention == True:
            out = self.mvdfe(out)

        return out


class CWDR(nn.Module):
    def __init__(self, num_of_classes, in_channel=1):
        super(CWDR, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.block1 = BasicBlock(in_ch=64, out_ch=64, stride=1, attention=True, attention_kernel_size=7)
        self.block2 = BasicBlock(in_ch=64, out_ch=64, stride=1, attention=True, attention_kernel_size=7)
        self.block3 = BasicBlock(in_ch=64, out_ch=64, stride=1, attention=True, attention_kernel_size=7)

        self.ac = Class_Specific_Classifier(input_dim=64, num_classes=num_of_classes)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.ac(x)

        x = self.sigmoid(x)

        return x