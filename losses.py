import paddle
import paddle.nn as nn
import numpy as np

class ReconstructionLoss(nn.Layer):
    def __init__(self, losstype='l2', eps=1e-3):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target):
        if self.losstype == 'l2':
            return paddle.mean(paddle.sum((x - target)**2, (1, 2, 3)))
        elif self.losstype == 'l1':
            diff = x - target
            return paddle.mean(paddle.sum(paddle.sqrt(diff * diff + self.eps), (1, 2, 3)))
        elif self.losstype == 'l_log':
            diff = x - target
            eps = 1e-6
            return paddle.mean(paddle.sum(-paddle.log(1-diff.abs()+eps), (1, 2, 3)))
        else:
            print("reconstruction loss type error!")
            return 0

# Gradient Loss
class Gradient_Loss(nn.Layer):
    def __init__(self, losstype='l2'):
        super(Gradient_Loss, self).__init__()
        a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        conv1 = nn.Conv2D(3, 3, kernel_size=3, stride=1, padding=1, bias_attr=False, groups=3)
        a = paddle.to_tensor(a, dtype="float32").unsqueeze(0)
        a = paddle.stack((a, a, a))
        conv1.weight = paddle.create_parameter(shape=a.shape,
                                dtype=str(a.numpy().dtype),
                                default_initializer=paddle.nn.initializer.Assign(a))
        conv1.weight.stop_gradient = True
        self.conv1 = conv1

        b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        conv2 = nn.Conv2D(3, 3, kernel_size=3, stride=1, padding=1, bias_attr=False, groups=3)
        b = paddle.to_tensor(b, dtype="float32").unsqueeze(0)
        b = paddle.stack((b, b, b))
        conv2.weight = paddle.create_parameter(shape=b.shape,
                                dtype=str(b.numpy().dtype),
                                default_initializer=paddle.nn.initializer.Assign(b))
        conv2.weight.stop_gradient = True
        self.conv2 = conv2

        self.Loss_criterion = nn.L1Loss()

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        y1 = self.conv1(y)
        y2 = self.conv2(y)

        l_h = self.Loss_criterion(x1, y1)
        l_v = self.Loss_criterion(x2, y2)
        return l_h + l_v #+ l_total


class SSIM_Loss(nn.Layer):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM_Loss, self).__init__()
        self.mu_x_pool   = nn.AvgPool2D(3, 1)
        self.mu_y_pool   = nn.AvgPool2D(3, 1)
        self.sig_x_pool  = nn.AvgPool2D(3, 1)
        self.sig_y_pool  = nn.AvgPool2D(3, 1)
        self.sig_xy_pool = nn.AvgPool2D(3, 1)

        self.refl = nn.Pad2D(1, mode="reflect")

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return paddle.clip((1 - SSIM_n / SSIM_d) / 2, 0, 1)

if __name__ == '__main__':
    l1 = ReconstructionLoss()
    l2 = Gradient_Loss()
    l3 = SSIM_Loss()
    a,b = paddle.ones((1,3,10,10)),paddle.ones((1,3,10,10))
    r1,r2,r3 = l1(a,b), l2(a,b), l3(a,b)
    print("r1",r1)
    print("r2",r2)
    print("r3",r3.shape)