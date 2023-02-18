from torch import nn


class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, relu=True, bn=True):
        super(BasicConv1D, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Class_Specific_Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Class_Specific_Classifier, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.conv1 = BasicConv(input_dim, num_classes, 1, stride=1, padding=0, relu=False, bn=False)  # bn must be False here
        self.conv2 = BasicConv1D(num_classes, 4, 3, stride=1, padding=1, relu=True, bn=True)
        self.conv3 = BasicConv1D(4, num_classes, 3, stride=1, padding=1, relu=True, bn=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = x.flatten(2)
        identity = x

        scale = self.conv2(x)
        scale = self.conv3(scale)
        scale = self.sigmoid(scale)

        out = identity * scale
        out = self.avgpool(out)
        out = out.squeeze(-1)

        return out
