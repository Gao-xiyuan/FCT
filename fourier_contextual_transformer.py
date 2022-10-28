import torch
import torch.nn as nn


class FCT(nn.Module):
    def __init__(self, dim=2048, decode_dim=1024, hw=400):
        super(FCT, self).__init__()
        self.dim_o = dim
        a = dim
        dim, decode_dim = hw, hw
        hw = a
        self.decode_dim = decode_dim
        self.weight_q = nn.Linear(dim, decode_dim, bias=False)
        self.weight_k = nn.Linear(dim, decode_dim, bias=False)
        self.weight_alpha = nn.Parameter(torch.randn(hw // 2 + 1, hw // 2 + 1) * 0.02)
        self.proj = nn.Linear(hw, hw)
        self.ac_bn_2 = torch.nn.Sequential(torch.nn.ReLU(), nn.BatchNorm2d(self.dim_o))
        self.writer = open('../../nan_check.txt', 'w')

    def forward(self, x):  # 【B，C，N】
        raw = x
        B, C, H, W = x.shape
        N = H * W
        x = x.reshape(B, C, N)  # .transpose(-2, -1) #[B, N, C]
        q = self.weight_q(x).transpose(-2, -1)  # [B，N，C]
        k = self.weight_k(x).transpose(-2, -1)  # [B，N，C]
        q = torch.fft.rfft2(q, dim=(-2, -1), norm='ortho')
        k = torch.fft.rfft2(k, dim=(-2, -1), norm='ortho')

        '''
        [B,N,C//2+1]
        '''
        q_r, q_i = q.real.transpose(-2, -1), q.imag.transpose(-2, -1)
        attn_r = q_r @ k.real  # [N,N]
        attn_i = q_i @ k.imag  # [N,N]
        attn_r = self.weight_alpha * attn_r
        attn_i = self.weight_alpha * attn_i
        x_r = logmax(attn_r) @ q_i  # [B, N, C] 无softmax 95.7
        x_i = logmax(attn_i) @ q_r  # [B, N, C]
        x = torch.view_as_complex(torch.stack([x_r, x_i], dim=-1)).transpose(-2, -1)
        x = torch.fft.irfft2(x, dim=(-2, -1), norm='ortho')
        x = self.proj(x)
        x = x.reshape(B, C, H, W)
        x = self.ac_bn_2(x)
        return raw + x

def logmax(X, axis=-1):
    X_log = torch.log(X)
    return X_log / X_log.sum(axis, keepdim=True)