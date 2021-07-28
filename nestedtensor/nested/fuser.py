import torch


def is_convolutional_block(sequential_module):
    if len(sequential_module) != 7:
        return False
    return True
    print(sequential_module[0])
    print(type(sequential_module[0]))
    print(dir(sequential_module[0]))
    parameters = list(sequential_module[0].parameters())
    if len(parameters) != 1:
        return False
    print(parameters[0].size())
    print(sequential_module[0].state_dict())
    return True


def computeUpdatedConvWeightAndBias(
        bn_rv,
        bn_eps,
        bn_w,
        bn_b,
        bn_rm,
        conv_w,
        conv_b=None):
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
    new_w = conv_w * (bn_w * bn_var_rsqrt).reshape(-1, 1, 1, 1)
    if conv_b is None:
        return new_w
    new_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    return new_w, new_b


class NewModule(torch.nn.Module):

    def __init__(self, old_sequential_module):
        super(NewModule, self).__init__()
        # print("old_sequential_module")
        # print(old_sequential_module)
        self.layer0 = old_sequential_module[0]
        if self.layer0.bias is None:
            self.layer0_bias = torch.zeros((self.layer0.weight.size(0)),
                    dtype=self.layer0.weight.dtype, device=torch.device('cuda'))
        self.layer1 = old_sequential_module[1]
        self.layer2 = old_sequential_module[2]
        self.layer3 = old_sequential_module[3]
        self.layer4 = old_sequential_module[4]
        self.layer5 = old_sequential_module[5]
        self.layer6 = old_sequential_module[6]
        # print(list(self.layer0.named_parameters()))
        # print(list(self.layer1.named_parameters()))
        self.layer0.weight.data = computeUpdatedConvWeightAndBias(
                self.layer1.running_var,
                self.layer1.eps,
                self.layer1.weight,
                self.layer1.bias,
                self.layer1.running_mean,
                self.layer0.weight.data)
        self.layer3.weight.data = computeUpdatedConvWeightAndBias(
                self.layer4.running_var,
                self.layer4.eps,
                self.layer4.weight,
                self.layer4.bias,
                self.layer4.running_mean,
                self.layer3.weight.data)
        # import sys; sys.exit(1)

    def forward(self, inp):

        if False:
            inp2 = torch.cudnn_convolution_relu(inp,
                                                self.layer0.weight,
                                                self.layer0_bias,
                                                self.layer0.stride,
                                                self.layer0.padding,
                                                self.layer0.dilation,
                                                self.layer0.groups)
        else:
            inp1 = self.layer0(inp)  # Conv2d
            # # inp4 = self.layer1(inp3)  # BatchNorm2d
            inp2 = self.layer2(inp1) # Relu
        if False and self.layer3.groups == 1:
            inp5 = torch.cudnn_convolution_relu(inp2,
                                                self.layer3.weight,
                                                self.layer3.bias,
                                                self.layer3.stride,
                                                self.layer3.padding,
                                                self.layer3.dilation,
                                                self.layer3.groups)
        else:
            inp3 = self.layer3(inp2)  # Conv2d
            # # inp4 = self.layer4(inp3)  # BatchNorm2d
            inp5 = self.layer5(inp3) # Relu
        inp6 = self.layer6(inp5)
        return inp6


def fuse_convolutional_block(sequential_module):
    # print(sequential_module)
    conv2d_2 = sequential_module[3]
    conv2d_3 = sequential_module[6]
    assert conv2d_2.padding_mode == "zeros"
    assert conv2d_3.padding_mode == "zeros"
    return NewModule(sequential_module)


def _sequential_fuser(sequential_module):
    if is_convolutional_block(sequential_module):
        return fuse_convolutional_block(sequential_module)
    assert False
