import torch
import torch.nn as nn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAttention(nn.Module):

    def __init__(self, in_channels, reduction=32):
        super(CoordAttention, self).__init__()
        # AdaptiveAvgPool2d((1, None)):[n,c,1,w]    nn.AdaptiveAvgPool2d((None, 1):[n,c,h,1]
        # pool_w：[n,c,1,w]y方向的平均池化    pool_h: [n,c,h,1]x方向的平均池化
        self.pool_w, self.pool_h, self.pool_c = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1)), nn.AdaptiveAvgPool2d(1)
        self.max_w, self.max_h, self.max_c = nn.AdaptiveMaxPool2d((1, None)), nn.AdaptiveMaxPool2d((None, 1)), nn.AdaptiveMaxPool2d(1)
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, in_channels, kernel_size=(1,2), stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, in_channels, kernel_size=(2,1), stride=1, padding=0)
        self.conv4 = nn.Conv2d(temp_c, in_channels, kernel_size=(1,2), stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        # x_h:[n,c,h,1]    x_w:[n,c,1,w]   x_c:[n,c,1,1]
        x_h, x_w, x_c = self.pool_h(x), self.pool_w(x), self.pool_c(x)

        # m_h:[n,c,h,1]    m_w:[n,c,1,w]   m_c:[n,c,1,1]
        m_h, m_w, m_c = self.max_h(x), self.max_w(x), self.max_c(x) # n, c, H, W = x.shape

        # cat_h:[n,c,h,2]   cat_w:[n,c,2,w]->[n,c,w,2]    cat_c:[n,c,1,2]
        cat_h, cat_w, cat_c = torch.cat([x_h, m_h], dim=3), torch.cat([x_w, m_w], dim=2).permute(0, 1, 3, 2), torch.cat([x_c, m_c], dim=3)

        # x_cat:[n,c,h+w+1,2]
        x_cat = torch.cat([cat_h, cat_w, cat_c], dim=2)

        out = self.act1(self.bn1(self.conv1(x_cat)))

        # x_h:[n,c,h,2]   x_w:[n,c,w,2]   x_c:[n,c,1,2]
        x_h, x_w, x_c = torch.split(out, [H, W, 1], dim=2)
        # x_w:[n,c,2,w]
        x_w = x_w.permute(0, 1, 3, 2)
        # out_h:[n,c,h,2]
        out_h = torch.sigmoid(self.conv2(x_h))
        # out_w:[n,c,2,w]
        out_w = torch.sigmoid(self.conv3(x_w))
        # out_c:[n,c,1,2]
        out_c = torch.sigmoid(self.conv4(x_c))
        return short * out_w * out_h * out_c


# if __name__ == '__main__':
#     x = torch.randn(5, 16, 32, 32)  # b, c, h, w
#     ca_model = CoordAttention(16, 16)
#     y = ca_model(x)
#     print(y.shape)