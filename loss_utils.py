#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def l1_loss(network_output, gt):
    # return torch.abs((network_output - gt)).mean()
    return my_loss(network_output, gt)


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def my_loss(network_output, gt):
    return frequency_sensitive_loss(network_output, gt)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fft_image_split(image, low_freq_threshold=0.2):
    """
    Split an image into low and high frequency components using FFT.
    """
    # Add a batch dimension if it's not present
    original_dim = len(image.size())
    if original_dim == 3:
        image = image.unsqueeze(0)

    fft = torch.fft.fft2(image, dim=(-2, -1))
    fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))

    _, _, rows, cols = image.size()
    crow, ccol = rows // 2, cols // 2

    # Create a mask with a low frequency square set to 1 and the rest to 0
    mask = torch.zeros_like(image, dtype=torch.float)
    mask[:, :, crow - int(low_freq_threshold * rows) // 2:crow + int(low_freq_threshold * rows) // 2,
    ccol - int(low_freq_threshold * cols) // 2:ccol + int(low_freq_threshold * cols) // 2] = 1

    low_freq = torch.fft.ifft2(torch.fft.ifftshift(fft_shift * mask, dim=(-2, -1)), dim=(-2, -1)).real
    high_freq = torch.fft.ifft2(torch.fft.ifftshift(fft_shift * (1 - mask), dim=(-2, -1)), dim=(-2, -1)).real

    # Remove the batch dimension if it was originally a 3D tensor
    if original_dim == 3:
        low_freq = low_freq.squeeze(0)
        high_freq = high_freq.squeeze(0)

    return low_freq, high_freq


def frequency_sensitive_loss(network_output, gt, low_freq_threshold=100):
    """
    Compute the loss, separately for low and high frequency components of the image.
    """

    network_output_low, network_output_high = fft_image_split(network_output, low_freq_threshold)
    gt_low, gt_high = fft_image_split(gt, low_freq_threshold)

    # Compute losses for low and high frequency components
    loss_low = F.mse_loss(network_output_low, gt_low)
    loss_high = F.mse_loss(network_output_high, gt_high)

    # Combine losses
    total_loss = loss_low + loss_high  # You can also weight these losses differently if needed

    return total_loss
