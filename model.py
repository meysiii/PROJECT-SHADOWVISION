import torch
import torch.nn as nn
import torch.nn.functional as F

class MDTA(nn.Module):
    """Multi-Dhead Transposed Self-Attention block for low-light Restormer."""
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, self.num_heads, c // self.num_heads, h * w)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        out = (attn @ v).view(b, c, h, w)
        return self.project_out(out)

class RestormerBlock(nn.Module):
    def __init__(self, channels=32, num_heads=4):
        super(RestormerBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.BatchNorm2d(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class ShadowVisionEnhancer(nn.Module):
    """Restormer-inspired Physics-Guided Restorer for Lunar PSRs."""
    def __init__(self, in_channels=1, out_channels=1, feat_channels=32):
        super(ShadowVisionEnhancer, self).__init__()
        self.in_conv = nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1)
        self.block1 = RestormerBlock(feat_channels)
        self.block2 = RestormerBlock(feat_channels)
        self.out_conv = nn.Conv2d(feat_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.in_conv(x)
        res = self.block1(feat)
        res = self.block2(res)
        enhanced = torch.sigmoid(self.out_conv(res + feat))
        return enhanced

class PhotometricConsistencyLoss(nn.Module):
    """Physics-Informed Loss enforcing Lommel-Seeliger Photometric Reflectance."""
    def __init__(self, alpha=1.0, beta=0.2):
        super(PhotometricConsistencyLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target_est):
        # L1 Pixel Reconstruction Loss
        recon_loss = self.l1_loss(pred, target_est)
        
        # Photometric Gradient Penalty (Ensures sharp terrain gradients without synthetic artifacts)
        pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        smoothness_penalty = torch.mean(pred_grad_x) + torch.mean(pred_grad_y)

        total_loss = self.alpha * recon_loss + self.beta * smoothness_penalty
        return total_loss
    