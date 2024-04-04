import torch
import torch.nn as nn


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

        L2_distance_square = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance_square) / (n_samples ** 2 - n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        # print(bandwidth_list)

        kernel_val = [torch.exp(-L2_distance_square / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return sum(kernel_val)  # /len(kernel_val)

    def forward(self, source, target, source_mask, target_mask, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

        source = source.view(size=(-1, 128))  # 此处原来是128
        source_mask = source_mask.view(size=(-1,))
        # print("source ", source.shape)
        # print("source ", source_mask.shape)
        source = source[source_mask.bool()]
        # print("source", source.shape)

        target = target.view(size=(-1, 128))  # 此处原来是128
        target_mask = target_mask.view(size=(-1,))
        # print("target ", target.shape)
        # print("target ", target_mask.shape)
        target = target[target_mask.bool()]
        # print("target", target.shape)

        source_num = int(source.size()[0])
        target_num = int(target.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num,
                                       fix_sigma=fix_sigma)
        XX = torch.mean(kernels[:source_num, :source_num])
        YY = torch.mean(kernels[source_num:, source_num:])
        XY = torch.mean(kernels[:source_num, source_num:])
        YX = torch.mean(kernels[source_num:, :source_num])
        loss = XX + YY - XY - YX
        return loss