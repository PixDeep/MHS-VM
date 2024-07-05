import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from scanutils import *

def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
    

class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim*2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x
    

class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x


class VMMBV0(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        bias=False,
        device=None,
        dtype=None,
        scan_type='C', # Z, C, X, S
        scan_routes=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.expand = expand
        self.d_inner = self.d_model # int(self.expand * self.d_model) # 
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.scan_type = scan_type
        self.scan_routes = scan_routes
        self.selective_scan = selective_scan_fn
        
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    # xs --mamba2d--> ys
    def xs2ys(self, xs: torch.Tensor):

        b, k, d, l = xs.shape
        ## print('xs.shape: ', xs.shape, 'weight.shape: ', self.x_proj_weight.shape)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(b, k, -1, l), self.x_proj_weight)
        # 
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(b, k, -1, l), self.dt_projs_weight)
        # 
        xs = xs.float().view(b, -1, l) # (b, k * d, l)
        dts = dts.contiguous().float().view(b, -1, l) # (b, k * d, l)
        Bs = Bs.float().view(b, k, -1, l) # (b, k, d_state, l)
        Cs = Cs.float().view(b, k, -1, l) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_ys = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(b, k, -1, l)
        
        
        assert out_ys.dtype == torch.float
        return out_ys


    # scantype: C
    def x2ys_CScan(self, x: torch.Tensor):

        b, d, h, w = x.shape
        k = 4
        l = h*w

        # x -> xs

        ## scan_0 
        scan_tind_0 = self.scan_routes[0, 0]
        x_scan1d_0 = re_arrange(x, scan_tind_0)
        x_scan1d_0_inv = torch.flip(x_scan1d_0, dims=[-1])

        ## scan_1
        scan_tind_1 = self.scan_routes[1, 0]
        x_scan1d_1 = re_arrange(x, scan_tind_1)
        x_scan1d_1_inv = torch.flip(x_scan1d_1, dims=[-1])

        # xs
        xs = torch.stack([x_scan1d_0, x_scan1d_0_inv, x_scan1d_1, x_scan1d_1_inv], dim=1).view(b, k, -1, l)

        # xs --> ys
        ## print('in CScan')
        ys_mam = self.xs2ys(xs)
        
        # restore and reshape

        ## scan_0
        scan_pind_0 = self.scan_routes[0, 1]
        scan_pind_1 = self.scan_routes[1, 1]
        y_repack_0 = torch.index_select(ys_mam[:,0,:], dim=-1, index=scan_pind_0)
        y_repack_0_inv = torch.index_select(torch.flip(ys_mam[:,1,:], dims=[-1]), dim=-1, index=scan_pind_0)
        y_repack_1 = torch.index_select(ys_mam[:,2,:], dim=-1, index=scan_pind_1)
        y_repack_1_inv = torch.index_select(torch.flip(ys_mam[:,3,:], dims=[-1]), dim=-1, index=scan_pind_1)

        ys_repack = torch.stack([y_repack_0, y_repack_0_inv, y_repack_1, y_repack_1_inv], dim=1).view(b, k, -1, l)
        return ys_repack 
    
    # scantype: X
    def x2ys_XScan(self, x: torch.Tensor):

        b, d, h, w = x.shape
        k = 4
        l = h*w

        # x -> xs

        ## scan_0
        scan_tind_0 = self.scan_routes[2, 0]
        x_scan1d_0 = re_arrange(x, scan_tind_0)
        x_scan1d_0_inv = torch.flip(x_scan1d_0, dims=[-1])

        ## scan_1
        scan_tind_1 = self.scan_routes[3, 0]
        x_scan1d_1 = re_arrange(x, scan_tind_1)
        x_scan1d_1_inv = torch.flip(x_scan1d_1, dims=[-1])

        # xs
        xs = torch.stack([x_scan1d_0, x_scan1d_0_inv, x_scan1d_1, x_scan1d_1_inv], dim=1).view(b, k, -1, l)

        # xs --> ys
        ## print('in XScan')
        ys_mam = self.xs2ys(xs)

        # restore and stack
        scan_pind_0 = self.scan_routes[2, 1]
        scan_pind_1 = self.scan_routes[3, 1]
        y_repack_0 = torch.index_select(ys_mam[:,0,:], dim=-1, index=scan_pind_0)
        y_repack_0_inv = torch.index_select(torch.flip(ys_mam[:,1,:], dims=[-1]), dim=-1, index=scan_pind_0)
        y_repack_1 = torch.index_select(ys_mam[:,2,:], dim=-1, index=scan_pind_1)
        y_repack_1_inv = torch.index_select(torch.flip(ys_mam[:,3,:], dims=[-1]), dim=-1, index=scan_pind_1)

        ys_repack = torch.stack([y_repack_0, y_repack_0_inv, y_repack_1, y_repack_1_inv], dim=1).view(b, k, -1, l)
        return ys_repack

    # scantype: S, Spiral Scan
    def x2ys_SScan(self, x: torch.Tensor):

        b, d, h, w = x.shape
        k = 4
        l = h*w
        
        # x -> xs

        ## scan_0
        scan_tind_0 = self.scan_routes[4, 0]
        x_scan1d_0 = re_arrange(x, scan_tind_0)
        x_scan1d_0_inv = torch.flip(x_scan1d_0, dims=[-1])

        ## scan_1
        # upside-down
        scan_tind_1 = self.scan_routes[5, 0]
        x_scan1d_1 = re_arrange(x, scan_tind_1)
        x_scan1d_1_inv = torch.flip(x_scan1d_1, dims=[-1])

        # xs
        xs = torch.stack([x_scan1d_0, x_scan1d_0_inv, x_scan1d_1, x_scan1d_1_inv], dim=1).view(b, k, -1, l)

        # xs --> ys
        ## print('in SScan')
        ys_mam = self.xs2ys(xs)
        
        # repack
        scan_pind_0 = self.scan_routes[4, 1]
        scan_pind_1 = self.scan_routes[5, 1]
        y_repack_0 = torch.index_select(ys_mam[:,0,:], dim=-1, index=scan_pind_0)
        y_repack_0_inv = torch.index_select(torch.flip(ys_mam[:,1,:], dims=[-1]), dim=-1, index=scan_pind_0)
        y_repack_1 = torch.index_select(ys_mam[:,2,:], dim=-1, index=scan_pind_1)
        y_repack_1_inv = torch.index_select(torch.flip(ys_mam[:,3,:], dims=[-1]), dim=-1, index=scan_pind_1)

        ys_repack = torch.stack([y_repack_0, y_repack_0_inv, y_repack_1, y_repack_1_inv], dim=1).view(b, k, -1, l)

        return ys_repack

    def forward(self, x: torch.Tensor, **kwargs):

        b, d, h, w = x.shape
 
        if self.scan_type == 'C':
            ys = self.x2ys_CScan(x)
        elif self.scan_type == 'X':
            ys = self.x2ys_XScan(x)
        elif self.scan_type == 'S':
            ys = self.x2ys_SScan(x)
        assert ys.dtype == torch.float32
        
        return ys



# scan-section fusion
class SSF(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        
        padding = (kernel_size-1)//2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        
    def forward(self, x):
        
        # x.shape: b, k, d, l
        x_avg   = torch.mean(x, dim=1, keepdim=True)
        x_max,_ = torch.max(x, dim=1, keepdim=True)

        x_cat = torch.cat([x_avg, x_max], dim=1)
        x_cov = self.conv(x_cat)

        y = x_cov.squeeze(dim=1)

        return y  
    
# scan-section fusion
class SSF1x1(nn.Module):
    def __init__(self, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, bias=False)

    def forward(self, x):
        
        # x.shape: b, k, d, l
        x_avg   = torch.mean(x, dim=1, keepdim=True)
        x_max,_ = torch.max(x, dim=1, keepdim=True)

        x_cat = torch.cat([x_avg, x_max], dim=1)
        x_cov = self.conv(x_cat)
        x = x_cov.squeeze(dim=1)

        return x

class SSFCV_reluv0(nn.Module):
    def __init__(self):
        super().__init__()
        #self.t = t
        #self.conv = nn.Conv2d(3, 1, kernel_size, bias=False)
        self.act = nn.ReLU() # nn.Sigmoid() # Sigmoid()
        
    def forward(self, x):
        
        # x.shape: b, k, d, l
        x_sum  = torch.sum(x, dim=1, keepdim=True)
        x_std   = torch.std(x, dim=1, keepdim=True)
        x_min,_ = torch.min(x, dim=1, keepdim=True)
        # x_max,_ = torch.max(x, dim=1, keepdim=True)
        x_new =  x - x_min
        x_new_mean = torch.mean(x_new, dim=1, keepdim=True)
        x_cv = x_std / (x_new_mean + 1e-5)

        #t = 0.3
        scale = self.act(x_cv - 0.3) # *self.s # t=0.5, s=2 x_mean is ok 0.7995
        x_cov = x_sum * scale #self.conv(x_cat) * scale
        x = x_cov.squeeze(dim=1)
        # x.shape: b, d, l
        return x     

class SSFCV_relu_tp(nn.Module):
    def __init__(self):
        super().__init__()

        #self.t = nn.Parameter(0.6*torch.ones(1))
        self.act = nn.ReLU() 
        
    def forward(self, x):
        
        # x.shape: b, k, d, l
        x_sum  = torch.sum(x, dim=1, keepdim=True)
        x_std   = torch.std(x, dim=1, keepdim=True)
        x_min,_ = torch.min(x, dim=1, keepdim=True)
        #x_max,_ = torch.max(x, dim=1, keepdim=True)
        # x_range = x_max - x_min
        x_new =  x - x_min
        x_new_mean = torch.mean(x_new, dim=1, keepdim=True)
        x_cv = x_std / (x_new_mean + 1e-5)

        scale = self.act(x_cv - 0.5) # *self.s # t=0.5, s=2 x_mean is ok 0.7995
        x_cov = x_sum * scale 
        x = x_cov.squeeze(dim=1)
        # x.shape: b, d, l
        return x      
     
class PEEM(nn.Module):
    def __init__(
        self,
        d_model,
        head_num = 3,
        drop_rate: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        bias = False,
        d_conv =3,
        conv_bias = True,
        d_state: int = 16,
        device = None,
        dtype = None,
        dropout = 0,
        scan_routes = None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.ln_1 = norm_layer(d_model) # (hidden_dim)
        self.d_model = d_model 
        self.d_inner = d_model
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        
        # SS2D-Trident
        
        ## project to subspace
        self.head_num = head_num
        # dim of subspace
        self.d_sub = self.d_model//3
        # W_Z, W_X, W_S
        self.x_ssproj = nn.Linear(self.d_inner, self.d_sub*self.head_num, bias=False)
        self.x_ssproj_Ws = self.x_ssproj.weight 
        
        ## n scan-heads
        #self.mamba_Z = VMMBV0(d_model=self.d_sub, dropout=drop_rate, d_state=d_state, scan_type='Z', scan_routes = scan_routes, **kwargs)
        self.mamba_C = VMMBV0(d_model=self.d_sub, dropout=drop_rate, d_state=d_state, scan_type='C', scan_routes = scan_routes, **kwargs)
        self.mamba_X = VMMBV0(d_model=self.d_sub, dropout=drop_rate, d_state=d_state, scan_type='X', scan_routes = scan_routes, **kwargs)
        self.mamba_S = VMMBV0(d_model=self.d_sub, dropout=drop_rate, d_state=d_state, scan_type='S', scan_routes = scan_routes, **kwargs)

        self.ssf_C = SSFCV_relu_tp( ) 
        self.ssf_X = SSFCV_relu_tp( ) 
        self.ssf_S = SSFCV_relu_tp( ) 
        
        self.drop_path = DropPath(drop_path)
        
        print('in PEEM', scan_routes.shape)
        
        # after SS2D
        self.cat_norm = nn.LayerNorm(self.head_num*self.d_sub)
        #self.y_proj = nn.Linear(self.head_num*self.d_sub, self.d_model, bias=False)
       
        #self.out_norm = nn.LayerNorm(self.d_model)
        # self.out_proj = nn.Linear(self.d_model, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        
    def forward(self, x):
        # before SS2Ds
        b, h, w, c = x.shape
        xz = self.in_proj(x)

        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)

        # SS2D-Trident
        b, d, h, w = x.shape
        # 
        x_CXS = torch.einsum("b d h w, c d -> b c h w", x, self.x_ssproj_Ws)

        # split to X_C, X_X, X_S
        # x_C, x_X, x_S = rearrange(x_CXS, 'b (k d) h w -> k b d h w', k=n)
        x_C, x_X, x_S = torch.split(x_CXS, split_size_or_sections=self.d_sub, dim=1)

        # scan in subspaces
        ys_C = self.mamba_C(x_C)
        ys_X = self.mamba_X(x_X)
        ys_S = self.mamba_S(x_S)
        
        y_C  = self.ssf_C(ys_C)
        y_X  = self.ssf_X(ys_X)        
        y_S  = self.ssf_S(ys_S)
        
        # concatenate, norm and project
        y_cat = torch.cat([y_C, y_X, y_S], dim=1)
        y_cat_norm = self.cat_norm(y_cat.transpose(-1,-2)).contiguous() # b, l, c
        #y_ = self.y_proj(y_cat_norm) # b, l, d
       
        # after SS2Ds

        y = y_cat_norm.view(b, h, w, -1)
        #y = self.out_norm(y)
       
        y = y * F.silu(z)
        out = y 
        if self.dropout is not None:
            out = self.dropout(out)
        return out
        

class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        scan_routes = None,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)

        self.self_attention = PEEM(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, scan_routes=scan_routes, **kwargs)
        # self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        scan_routes=None,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        #print('in VSSLayer', scan_routes.shape)
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                scan_routes=scan_routes,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x
    


class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        upsample=None, 
        use_checkpoint=False, 
        d_state=16,
        scan_routes=None,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                scan_routes=scan_routes,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None


    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x
    


class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, scanroute_dict = None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        #print('in VSSM', scanroute_dict[0]['layer_0.cxs_tpinds'])
        # WASTED absolute position embedding ======================
        self.ape = False
        # self.ape = False
        # drop_rate = 0.0
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                scan_routes = scanroute_dict[i_layer]['layer_'+str(i_layer)+'.cxs_tpinds'].to('cuda'),
            )
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        num_layers = len(depths)-1
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
                scan_routes = scanroute_dict[num_layers-i_layer]['layer_'+str(num_layers-i_layer)+'.cxs_tpinds'].to('cuda'),
            )
            self.layers_up.append(layer)

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1]//4, num_classes, 1)

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        skip_list = []
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
        return x, skip_list
    
    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x+skip_list[-inx])

        return x
    
    def forward_final(self, x):
        x = self.final_up(x)
        x = x.permute(0,3,1,2)
        x = self.final_conv(x)
        return x

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x, skip_list = self.forward_features(x)
        x = self.forward_features_up(x, skip_list)
        x = self.forward_final(x)
        
        return x




    


