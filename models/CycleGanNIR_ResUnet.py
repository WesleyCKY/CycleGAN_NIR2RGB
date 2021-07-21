###############################################################################
# Combine ResNet34 with Unet256 architecture
###############################################################################

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision.models as models

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    num_blocks = get_norm_layer(norm_type=norm)
    norm_layer = get_norm_layer(norm_type=norm)

    #if netG == 'resnet_9blocks':
     # net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    #elif netG == 'resnet_6blocks':
        #net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    # elif netG == 'unet_128':
    #     net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    # elif netG == 'unet_256':
    #     net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    
    if netG == 'ResNet':
         net = ResNet(input_nc, output_nc, ngf, num_blocks, use_dropout=use_dropout)
    elif netG == 'ResNet34Unet':
        net = ResNet34Unet(input_nc, output_nc, ngf)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_net(net, init_type, init_gain, gpu_ids)


def define_D_1(input_nc, ndf, kernal_size, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        kernal_size(int)   -- the kernal size
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'DecoderBlock':  # default PatchGAN classifier
        net = DecoderBlock(input_nc, ndf, kernal_size, is_deconv=False, up_kwargs=None)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)

##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


# ======================================================================================================================
# =============================================   Resnet Generator     =================================================
# ======================================================================================================================

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

     adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


# ================================================================================


class DecoderBlock(nn.Module):
    def __init__(self,
                 input_nc=512,
                 ndf=256,
                 kernel_size=3,
                 is_deconv=False,
                 up_kwargs=None
                 ):
        super().__init__()
        self.up_kwargs = up_kwargs
        if kernel_size == 3:
            conv_padding = 1
        elif kernel_size == 1:
            conv_padding = 0
        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(input_nc,
                               input_nc // 4,
                               kernel_size,
                               padding=1,bias=False)
        self.norm1 = nn.BatchNorm2d(input_nc // 4)
        self.relu1 = nn.ReLU(inplace=True)
        # B, C/4, H, W -> B, C/4, H, W
        if is_deconv == True:
            self.deconv2 = nn.ConvTranspose2d(input_nc // 4,
                                              input_nc // 4,
                                              3,
                                              stride=2,
                                              padding=1,
                                              output_padding=conv_padding,bias=False)
        else:
            self.deconv2 = nn.Upsample(scale_factor=2)
        self.norm2 = nn.BatchNorm2d(input_nc // 4)
        self.relu2 = nn.ReLU(inplace=True)
        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(input_nc // 4,
                               ndf,
                               kernel_size,
                               padding=conv_padding,bias=False)
        self.norm3 = nn.BatchNorm2d(ndf)
        self.relu3 = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,  # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# =============================================================================

class ResNet34Unet(nn.Module):
    def __init__(self,
                 num_classes,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 ):
        super().__init__()
 
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
 
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.base_size=512
        self.crop_size=512
        self._up_kwargs={'mode': 'bilinear', 'align_corners': True}
        # self.firstconv = resnet.conv1
        # assert num_channels == 3, "num channels not used now. to use changle first conv layer to support num channels other then 3"
        # try to use 8-channels as first input
        if num_channels == 3:
            self.firstconv = resnet.conv1 #transfer to numpy Tensor
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),bias=False)
 
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
 
        # Decoder
        self.center = DecoderBlock(input_nc=filters[3],
                                   ndf=filters[3],
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)
        self.decoder4 = DecoderBlock(input_nc=filters[3] + filters[2],
                                     ndf=filters[2],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(input_nc=filters[2] + filters[1],
                                     ndf=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(input_nc=filters[1] + filters[0],
                                     ndf=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(input_nc=filters[0] + filters[0],
                                     ndf=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
 
        self.finalconv = nn.Sequential(nn.Conv2d(filters[0], 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.1, False),
                                       nn.Conv2d(32, num_classes, 1))
 
    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3,
                  self.encoder4]
 
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad
 
    def forward(self, x):
        # stem
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)
 
        # Encoder
        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
 
        center = self.center(e4)
 
        d4 = self.decoder4(torch.cat([center, e3], 1))
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        d1 = self.decoder1(torch.cat([d2, x], 1))
 
        f= self.finalconv(d1)
        return tuple([f])

# =======================   discriminator ====================================

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)



# # ======================   basic structure of CNN   =========================
# def double_conv(self, input_nc, output_nc):
#     return nn.Sequential(
#         nn.Conv2d(input_nc, output_nc, 3, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(output_nc, output_nc, 3, padding=1),
#         nn.ReLU(inplace=True)
#     )

# # ======================   Resnet block  =========================
# class BasicBlock(nn.Module):
#     """Basic Block for resnet 34
#     """
    
#     expansion = 1

#     def __init__(self, input_nc, output_nc, stride=1):
#         super().__init__()

#         #residual function
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(output_nc),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(output_nc, output_nc * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(output_nc * BasicBlock.expansion)
#         )

#         #shortcut
#         self.shortcut = nn.Sequential()

#         #the shortcut output dimension is not the same with residual function
#         #use 1*1 convolution to match the dimension
#         if stride != 1 or input_nc != BasicBlock.expansion * output_nc:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(input_nc, output_nc * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(output_nc * BasicBlock.expansion)
#             )
        
#     def forward(self, x):
#         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

# # =====================    Resnet generator  ========================

# class ResNet(nn.Module):

#     def __init__(self, input_nc, output_nc, use_dropout=False, num_block):

#         super().__init__()

#         # self.input_nc = 64

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(input_nc, 64, kernel_size = 7, stride = 2, padding = 3,bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True))
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         # we use a different inputsize than the original paper
#         # so conv2_x's stride is 1
#         self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
#         self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
#         self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
#         self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
#         # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         # self.fc = nn.Linear(512 * block.expansion, num_classes)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.dconv_up3 = double_conv(256 + 512, 256)
#         self.dconv_up2 = double_conv(128 + 256, 128)
#         self.dconv_up1 = double_conv(128 + 64, 64)

#         self.dconv_last=nn.Sequential(
#             nn.Conv2d(128, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(64,output_nc,1)
#         )

#     def _make_layer(self, block, output_nc, num_layers=nn.BatchNorm2d, stride):
#         """
#         Args:
#             block: block type, basic block or bottle neck block
#             output_nc: output depth channel number of this layer
#             num_blocks: how many blocks per layer
#             stride: the stride of the first block of this layer

#         Return:
#             return a resnet layer
#         """

#         # we have num_block blocks per layer, the first block
#         # could be 1 or 2, other blocks would always be 1
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.input_nc, output_nc, stride))
#             self.input_nc = output_nc * block.expansion

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         conv1 = self.conv1(x)
#         temp=self.maxpool(conv1)
#         conv2 = self.conv2_x(temp)
#         conv3 = self.conv3_x(conv2)
#         conv4 = self.conv4_x(conv3)
#         bottle = self.conv5_x(conv4)
        
#         # output = self.avg_pool(output)
#         # output = output.view(output.size(0), -1)
#         # output = self.fc(output)
        
#         x = self.upsample(bottle)
        
#         # print(x.shape)
#         # print(conv4.shape)
        
#         x = torch.cat([x, conv4], dim=1)

#         x = self.dconv_up3(x)
#         x = self.upsample(x)
        
#         # print(x.shape)
#         # print(conv3.shape)
        
#         x = torch.cat([x, conv3], dim=1)

#         x = self.dconv_up2(x)
#         x = self.upsample(x)
        
#         # print(x.shape)
#         # print(conv2.shape)
        
#         x = torch.cat([x, conv2], dim=1)

#         x = self.dconv_up1(x)
#         x=self.upsample(x)
        
#         # print(x.shape)
#         # print(conv1.shape)
        
#         x=torch.cat([x,conv1],dim=1)
#         self.model = self.dconv_last(x)

#         return self.model

#     def load_pretrained_weights(self):

#         model_dict=self.state_dict()
#         resnet34_weights = model.resnet34(True).state_dict()
#         count_res = 0
#         count_my = 0

#         reskeys = list(resnet34_weights.keys())
#         mykeys = list(model_dict.keys())
        
#         # print(self)
#         # print(models.resnet34())
#         # print(reskeys)
#         # print(mykeys)

#         corresp_map = []
#         while (True):              # list all with same suffix
#             reskey = reskeys[count_res]
#             mykey = mykeys[count_my]

#             if "fc" in reskey:
#                 break

#             while reskey.split(".")[-1] not in mykey:
#                 count_my += 1
#                 mykey = mykeys[count_my]

#             corresp_map.append([reskey, mykey])
#             count_res += 1
#             count_my += 1

#         for k_res, k_my in corresp_map:
#             model_dict[k_my]=resnet34_weights[k_res]

#         try:
#             self.load_state_dict(model_dict)
#             print("Loaded weight of resnet34 !")
#         except:
#             print("Error loading resnet34 weight !")
#             raise

# # ======================   BottleNeck   =========================

# # class BottleNeck(nn.Module):
# #     """Residual block for resnet over 50 layers
# #     """
# #     expansion = 4

# #     def __init__(self, input_nc, output_nc, stride=1):
# #         super().__init__()
# #         self.residual_function = nn.Sequential(
# #             nn.Conv2d(input_nc, output_nc, kernel_size=1, bias=False),
# #             nn.BatchNorm2d(output_nc),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(output_nc, output_nc, stride=stride, kernel_size=3, padding=1, bias=False),
# #             nn.BatchNorm2d(output_nc),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(output_nc, output_nc * BottleNeck.expansion, kernel_size=1, bias=False),
# #             nn.BatchNorm2d(output_nc * BottleNeck.expansion),
# #         )

# #         self.shortcut = nn.Sequential()

# #         if stride != 1 or input_nc != output_nc * BottleNeck.expansion:
# #             self.shortcut = nn.Sequential(
# #                 nn.Conv2d(input_nc, output_nc * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
# #                 nn.BatchNorm2d(output_nc * BottleNeck.expansion)
# #             )

# #     def forward(self, x):
# #         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

# # ============================= restnet 34 =======================================
# def resnet34(input_nc,output_nc,pretrain=True):
#     """ return a ResNet 34 object
#     """
#     model = ResNet(input_nc,output_nc,BasicBlock, [3, 4, 6, 3])
#     if pretrain:
#         model.load_pretrained_weights()
#     return model

