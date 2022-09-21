"""
## Invertible Denoising Network: A Light Solution for Real Noise Removal
## Liu, Yang and Qin, Zhenyue and Anwar, Saeed and Ji, Pan and Kim, Dongwoo and Caldwell, Sabrina and Gedeon, Tom
## CVPR 2021
## Modified from InvDN (https://github.com/Yang-Liu1082/InvDN)
@article{liu2021invertible,
  title={Invertible Denoising Network: A Light Solution for Real Noise Removal},
  author={Liu, Yang and Qin, Zhenyue and Anwar, Saeed and Ji, Pan and Kim, Dongwoo and Caldwell, Sabrina and Gedeon, Tom},
  journal={arXiv preprint arXiv:2104.10546},
  year={2021}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(ResBlock, self).__init__()
        feature = 64
        self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
        # self.conv1 = nn.Conv2D(channel_in, feature, kernel_size=3, padding=1, bias_attr=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(feature, feature, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2D(feature, feature, kernel_size=3, padding=1, bias_attr=True)
        self.conv3 = nn.Conv2d((feature+channel_in), channel_out, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2D((feature+channel_in), channel_out, kernel_size=3, padding=1, bias_attr=True)
        # self.init()

    def forward(self, x):
        residual = self.relu1(self.conv1(x))
        residual = self.relu1(self.conv2(residual))
        input = torch.cat((x, residual), dim=1)
        # input = paddle.concat((x, residual), 1)
        out = self.conv3(input)
        return out

    def init(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight.data,mean=0.0,std=0.02)

class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        
    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        # x1 = paddle.slice(x, [1], [0], [self.split_len1])
        # x2 = paddle.slice(x, [1], [self.split_len1], [self.split_len1 + self.split_len2])

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            # self.s = self.clamp * (F.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
            # y2 = x2.multiply(paddle.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            # self.s = self.clamp * (F.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            # y2 = (x2 - self.G(x1)).divide(paddle.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)
        # return paddle.concat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
            # jac = paddle.sum(self.s)
        else:
            jac = -torch.sum(self.s)
            # jac = -paddle.sum(self.s)

        return jac / x.shape[0]

class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)
        # self.haar_weights = paddle.ones([4, 1, 2, 2])

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False
        # self.haar_weights = paddle.concat([self.haar_weights] * self.channel_in, 0)
        # self.haar_weights = paddle.create_parameter(shape=self.haar_weights.shape,
        #                         dtype=str(self.haar_weights.numpy().dtype),
        #                         default_initializer=paddle.nn.initializer.Assign(self.haar_weights))
        # self.haar_weights.stop_gradient = True


    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            # out = paddle.transpose(out, [0,2,1,3,4])

            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])

            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            # out = paddle.transpose(out, [0,2,1,3,4])
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])

            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)
            # return F.conv2d_transpose(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac

class InvNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], down_num=2):
        super(InvNet, self).__init__()

        operations = []

        current_channel = channel_in
        for i in range(down_num):
            b = HaarDownsampling(current_channel)
            operations.append(b)
            current_channel *= 4
            for j in range(block_num[i]):
                b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations.append(b)

        self.operations = nn.ModuleList(operations)
        # self.operations = nn.LayerList(operations)

    def forward(self, x, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)

        if cal_jacobian:
            return out, jacobian
        else:
            return out

def constructor(channel_in, channel_out):
    return ResBlock(channel_in, channel_out)

def gaussian_batch(dims):
    return torch.randn(tuple(dims))
    # return paddle.randn(tuple(dims))

if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    scale = 4
    down_num = int(math.log(scale, 2))
    netG = InvNet(channel_in=3, channel_out=3, subnet_constructor=constructor, block_num=[8, 8], down_num=down_num)
    # di = torch.load("./InvDN_ResUnit_x4.pth")
    # netG.load_state_dict(di)
    # print(netG.operations[1].F.conv1.weight)
    # print(netG.operations[1].F.conv1.bias)
    # exit()



    gaussian_scale = 1

    data = torch.ones(1,3,100,100)
    output = netG(x=data) #1 48 25 25
    # print(output)
    # exit()

    y_forw = torch.cat((output[:, :3, :, :], gaussian_scale * gaussian_batch(output[:, 3:, :, :].shape)), axis=1)
    fake_H = netG(x=y_forw, rev=True)

    img = fake_H.detach().cpu() #TODO 
    print(img)

    # print("SSSSSSSSSSSSS")
    # paddle.utils.run_check()

    # a = ResBlock(3,3)
    # print(a.conv1.weight.data)
    # data = torch.ones(1,3,1,1)
    # b = a(data)
    # print(b)