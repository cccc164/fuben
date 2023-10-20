import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torchviz
from torch.autograd import Variable


class ConvAutoencoder(nn.Module):
    def __init__(self, kernel_size, in_channels=1):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=kernel_size,
                               stride=1, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=kernel_size,
                               stride=1, padding=kernel_size // 2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=kernel_size,
                               stride=1, padding=kernel_size // 2)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        # self.upsample = nn.Upsample(scale_factor=2)
        self.pad = nn.ConstantPad2d(padding=(0, 1, 0, 1), value=0)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=kernel_size,
                               stride=1, padding=kernel_size // 2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=kernel_size,
                               stride=1, padding=kernel_size // 2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=kernel_size,
                                          stride=1, padding=kernel_size // 2)
        self.conv = nn.Conv2d(128, in_channels, 1, stride=1, padding=0)

    def forward(self, image):
        conv1 = self.conv1(image)  # 82x102x5 102x82x5
        relu1 = F.relu(conv1)  # 82x102x128 102x82x128
        pool1, indice1 = self.pool(relu1)  # 41x51x128 51x41x128

        conv2 = self.conv2(pool1)  # 41x51x64 51x41x64
        relu2 = F.relu(conv2)
        pool2, indice2 = self.pool(relu2)  # 21x26x64 26x21x64

        conv3 = self.conv3(pool2)  # 21x26x32 26x21x32
        relu3 = F.relu(conv3)
        # pool3 = self.pool(relu3)  # 13x10x32

        deconv1 = self.deconv1(relu3)  # 21x26x32 26x21x32
        up1 = self.unpool(deconv1, indices=indice2)
        up1_pad = self.pad(up1)
        up_relu1 = F.relu(up1_pad)

        deconv2 = self.deconv2(up_relu1)  # 21x26x32 26x21x32
        up2 = self.unpool(deconv2, indices=indice1)
        up_relu2 = F.relu(up2)

        deconv3 = self.deconv3(up_relu2)  # 21x26x32 26x21x32
        up_relu3 = F.relu(deconv3)

        logits = self.conv(up_relu3)
        logits = torch.sigmoid(logits)
        return logits

class ConvAutoencoderOne(nn.Module):
    def __init__(self, kernel_size, in_channels=1):
        super(ConvAutoencoderOne, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=kernel_size,
                               stride=1, padding=kernel_size // 2)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        self.uppool = nn.MaxUnpool2d(2, 2)
        self.pad = nn.ConstantPad2d(padding=(0, 1, 0, 1), value=0)

        # Decoder
        # self.deconv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size,
        #                        stride=1, padding=kernel_size // 2)
        self.deconv1 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=kernel_size,
                                 stride=1, padding=kernel_size // 2)
        self.conv = nn.Conv2d(16, in_channels, 1, stride=1, padding=0)


    def forward(self, image):
        conv1 = self.conv1(image)  # 82x102x5 102x82x5
        relu1 = F.relu(conv1)  # 82x102x128 102x82x128
        pool1, index = self.pool(relu1)  # 41x51x128 51x41x128

        up1 = self.uppool(pool1, indices=index)
        deconv1 = self.deconv1(up1)  # 21x26x32 26x21x32
        up_relu1 = F.relu(deconv1)

        logits = self.conv(up_relu1)
        logits = torch.sigmoid(logits)
        return logits


class Autoencoder(nn.Module):
    def __init__(self, in_channels):
        super(Autoencoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, padding=0)
        self.pool2 = nn.MaxPool2d(2, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.pad = nn.ConstantPad2d(padding=(0, 1, 0, 0), value=0)

        # Decoder
        self.up1 = nn.ConvTranspose2d(32, 32, 2, stride=2, padding=0, output_padding=0)
        self.up2 = nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(64, 128, 2, stride=2, padding=0)
        self.conv = nn.Conv2d(128, in_channels, 1, stride=1, padding=0)

    def forward(self, image):
        conv1 = self.conv1(image)  # 82x102x5 102x82x5
        relu1 = F.relu(conv1)  # 82x102x128 102x82x128
        pool1 = self.pool1(relu1)  # 41x51x128 51x41x128

        conv2 = self.conv2(pool1)  # 41x51x64 51x41x64
        relu2 = F.relu(conv2)
        pool2 = self.pool2(relu2)  # 21x26x64 26x21x64

        conv3 = self.conv3(pool2)  # 21x26x32 26x21x32
        relu3 = F.relu(conv3)
        pool3 = self.pool3(relu3)  # 13x10x32

        up1 = self.up1(pool3)  # 21x26x32 26x21x32
        up_relu1 = F.relu(up1)
        up_relu1 = self.pad(up_relu1)

        up2 = self.up2(up_relu1)  # 20x20x32
        up_relu2 = F.relu(up2)

        up3 = self.up3(up_relu2)
        up_relu3 = F.relu(up3)

        logits = self.conv(up_relu3)
        logits = torch.sigmoid(logits)
        return logits


class ConvAutoencoderSample(nn.Module):
    def __init__(self, kernel_size, in_channels, base=32):
        super(ConvAutoencoderSample, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=base*4, kernel_size=kernel_size,
                               stride=1, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(in_channels=base*4, out_channels=base*2, kernel_size=kernel_size,
                               stride=1, padding=kernel_size // 2)
        self.conv3 = nn.Conv2d(in_channels=base*2, out_channels=base, kernel_size=kernel_size,
                               stride=1, padding=kernel_size // 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.pad = nn.ConstantPad2d(padding=(0, 1, 0, 1), value=0)

        # Decoder
        self.deconv1 = nn.Conv2d(in_channels=base, out_channels=base, kernel_size=kernel_size,
                                 stride=1, padding=kernel_size // 2)
        self.deconv2 = nn.Conv2d(in_channels=base, out_channels=base*2, kernel_size=kernel_size,
                                 stride=1, padding=kernel_size // 2)
        self.deconv3 = nn.Conv2d(in_channels=base*2, out_channels=base*4, kernel_size=kernel_size,
                                 stride=1, padding=kernel_size // 2)
        self.conv = nn.Conv2d(base*4, in_channels, 5, stride=1, padding=2)

    def forward(self, image):
        conv1 = self.conv1(image)  # 82x102x5 102x82x5
        relu1 = F.relu(conv1)  # 82x102x128 102x82x128
        pool1 = self.pool(relu1)  # 41x51x128 51x41x128

        conv2 = self.conv2(pool1)  # 41x51x64 51x41x64
        relu2 = F.relu(conv2)
        pool2 = self.pool(relu2)  # 21x26x64 26x21x64

        conv3 = self.conv3(pool2)  # 21x26x32 26x21x32
        relu3 = F.relu(conv3)
        # pool3 = self.pool(relu3)  # 13x10x32

        deconv1 = self.deconv1(relu3)  # 21x26x32 26x21x32
        up1 = self.upsample(deconv1)
        up1_pad = self.pad(up1)
        up_relu1 = F.relu(up1_pad)

        deconv2 = self.deconv2(up_relu1)  # 21x26x32 26x21x32
        up2 = self.upsample(deconv2)
        up_relu2 = F.relu(up2)

        deconv3 = self.deconv3(up_relu2)  # 21x26x32 26x21x32
        up_relu3 = F.relu(deconv3)

        logits = self.conv(up_relu3)
        logits = torch.sigmoid(logits)
        return logits



class ConvAutoencoderSampleOne(nn.Module):
    def __init__(self, kernel_size, in_channels):
        super(ConvAutoencoderSampleOne, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=kernel_size,
                               stride=1, padding=kernel_size // 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.pad = nn.ConstantPad2d(padding=(0, 1, 0, 1), value=0)

        # Decoder
        self.deconv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=kernel_size,
                                 stride=1, padding=kernel_size // 2)
        self.conv = nn.Conv2d(128, in_channels, 5, stride=1, padding=2)

    def forward(self, image):
        conv1 = self.conv1(image)  # 82x102x5 102x82x5
        relu1 = F.relu(conv1)  # 82x102x128 102x82x128
        pool1 = self.pool(relu1)  # 41x51x128 51x41x128  # 21x26x64 26x21x64

        deconv1 = self.deconv1(pool1)  # 21x26x32 26x21x32
        up1 = self.upsample(deconv1)
        up_relu1 = F.relu(up1)

        logits = self.conv(up_relu1)
        logits = torch.sigmoid(logits)
        return logits



class ConvAutoencoderSampleTwo(nn.Module):
    def __init__(self, kernel_size, in_channels):
        super(ConvAutoencoderSampleTwo, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=kernel_size,
                               stride=1, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=kernel_size,
                               stride=1, padding=kernel_size // 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.pad = nn.ConstantPad2d(padding=(0, 1, 0, 1), value=0)

        # Decoder
        self.deconv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size,
                                 stride=1, padding=kernel_size // 2)
        self.deconv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size,
                                 stride=1, padding=kernel_size // 2)
        self.conv = nn.Conv2d(128, in_channels, 5, stride=1, padding=2)

    def forward(self, image):
        conv1 = self.conv1(image)  # 82x102x5 102x82x5
        relu1 = F.relu(conv1)  # 82x102x128 102x82x128
        pool1 = self.pool(relu1)  # 41x51x128 51x41x128

        conv2 = self.conv2(pool1)  # 41x51x64 51x41x64
        relu2 = F.relu(conv2)
        pool2 = self.pool(relu2)  # 21x26x64 26x21x64

        deconv1 = self.deconv1(pool2)  # 21x26x32 26x21x32
        up1 = self.upsample(deconv1)
        up1_pad = self.pad(up1)
        up_relu1 = F.relu(up1_pad)

        deconv2 = self.deconv2(up_relu1)  # 21x26x32 26x21x32
        up2 = self.upsample(deconv2)
        up_relu2 = F.relu(up2)

        logits = self.conv(up_relu2)
        logits = torch.sigmoid(logits)

        return logits
