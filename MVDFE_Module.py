import torch
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


class Pooling(nn.Module):
    def forward(self, x):
        max = torch.max(x, 1)[0].unsqueeze(1)
        avg = torch.mean(x, 1).unsqueeze(1)
        return torch.cat((max, avg), dim=1)


class Branch(nn.Module):
    def __init__(self, kernel_size=7):
        super(Branch, self).__init__()
        self.compress = Pooling()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class MVDFE_Module(nn.Module):
    def __init__(self, kernel_size=7):
        super(MVDFE_Module, self).__init__()
        self.HW_branch = Branch(kernel_size=kernel_size)
        self.CW_branch = Branch(kernel_size=kernel_size)
        self.HC_branch = Branch(kernel_size=kernel_size)


    def forward(self, x):
        # HW branch recalibrate
        HW_out = self.HW_branch(x)

        # CW branch recalibrate
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        CW_out = self.CW_branch(x_perm1)
        CW_out = CW_out.permute(0, 2, 1, 3).contiguous()

        # HC branch recalibrate
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        HC_out = self.HC_branch(x_perm2)
        HC_out = HC_out.permute(0, 3, 2, 1).contiguous()

        final_out = 1/3 * (HW_out + CW_out + HC_out)

        return final_out
