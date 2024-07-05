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


DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
        
    
from scanutils import *
    


# Patch Embedding Sequence Modeling Engine Based on Mamba    
class MambaEngine(nn.Module):
    r""" 
        Input:  xs | xs.shape  -->  (b, k, d, l).
        Output: ys | ys.shape  -->  (b, k, d, l).
    """    
    def __init__(
        self,
        d_model,
        d_state=16,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = self.d_model 
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

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

    def forward(self, xs: torch.Tensor): 

        b, k, d, l = xs.shape
        
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
    

# Patch Embedding Map Scan Pattern    
class ScanPattern(nn.Module):
    r""" 
        Input:  x,    x.shape  -->  (b, d, h, w).
        Output: xs,  xs.shape  -->  (b, k, d, l).
    """ 
    def __init__(
        self, 
        pattern='C', 
        s2s_engine=None,
        route_dict=None, 
        **kwargs
    ):
        super().__init__()
        
        self.scan_pattern    = pattern
        self.scan_route_dict = route_dict
        self.seq2seq_engine  = s2s_engine 
        if self.scan_pattern == 'C':
            self.scan_tind_0 = self.scan_route_dict[0, 0]
            self.scan_tind_1 = self.scan_route_dict[1, 0]
            self.scan_pind_0 = self.scan_route_dict[0, 1]
            self.scan_pind_1 = self.scan_route_dict[1, 1]            
        elif self.scan_pattern == 'X':
            self.scan_tind_0 = self.scan_route_dict[2, 0]
            self.scan_tind_1 = self.scan_route_dict[3, 0] 
            self.scan_pind_0 = self.scan_route_dict[2, 1]
            self.scan_pind_1 = self.scan_route_dict[3, 1]            
        elif self.scan_pattern == 'S':
            self.scan_tind_0 = self.scan_route_dict[4, 0]
            self.scan_tind_1 = self.scan_route_dict[5, 0]
            self.scan_pind_0 = self.scan_route_dict[4, 1]
            self.scan_pind_1 = self.scan_route_dict[5, 1]            
        elif self.scan_pattern == 'Z':
            self.scan_tind_0 = self.scan_route_dict[6, 0]
            self.scan_tind_1 = self.scan_route_dict[7, 0]
            self.scan_pind_0 = self.scan_route_dict[6, 1]
            self.scan_pind_1 = self.scan_route_dict[7, 1]            
            
    def ScanRoutes(self, x: torch.Tensor):    
        
        b, d, h, w = x.shape
        k = 4
        l = h*w
        
        x_scan1d_0     = index_select_2d(x, self.scan_tind_0)
        x_scan1d_1     = index_select_2d(x, self.scan_tind_1)
        x_scan1d_0_inv = torch.flip(x_scan1d_0, dims=[-1])
        x_scan1d_1_inv = torch.flip(x_scan1d_1, dims=[-1])
        
        xs = torch.stack([x_scan1d_0, x_scan1d_0_inv, x_scan1d_1, x_scan1d_1_inv], dim=1).view(b, k, -1, l)
        
        return xs
    
    def ReArrange(self, ys_s2s: torch.Tensor):
        
        b, k, d, l  = ys_s2s.shape
        
        y_re_0      = torch.index_select(ys_s2s[:,0], dim=-1, index=self.scan_pind_0)
        y_re_1      = torch.index_select(ys_s2s[:,2], dim=-1, index=self.scan_pind_1)        
        y_re_0_inv  = torch.index_select(torch.flip(ys_s2s[:,1], dims=[-1]), dim=-1, index=self.scan_pind_0)
        y_re_1_inv  = torch.index_select(torch.flip(ys_s2s[:,3], dims=[-1]), dim=-1, index=self.scan_pind_1)
        
        ys_re       = torch.stack([y_re_0, y_re_0_inv, y_re_1, y_re_1_inv], dim=1).view(b, k, -1, l)
        
        return ys_re

    def forward(self, x: torch.Tensor, **kwargs):

        xs     = self.ScanRoutes(x)
        ys_s2s = self.seq2seq_engine(xs)
        ys     = self.ReArrange(ys_s2s)
        
        assert ys.dtype == torch.float32
        
        return ys

    
# Embedding Section Fusion with Gated Fusion Unit (GFU)
class GFU(nn.Module):
    r""" 
        Input:  x,  x.shape  -->  (b, k, d, l).
        Output: y,  y.shape  -->  (b, d, l).
    """ 
    def __init__(self, t=0.0):
        super().__init__()
        
        self.t = t
        self.act = nn.ReLU() 
        
    def forward(self, x):
        
        x_sum   = torch.sum(x, dim=1, keepdim=True)
        x_std   = torch.std(x, dim=1, keepdim=True)
        x_min,_ = torch.min(x, dim=1, keepdim=True)
        x_      =  x - x_min
        x_mean  = torch.mean(x_, dim=1, keepdim=True)
        x_cv    = x_std / (x_mean + 1e-5)

        x_fs    = x_sum * self.act(x_cv-self.t)
        x       = x_fs.squeeze(dim=1)

        return x       

        
class ScanHead(nn.Module):
    r""" Scan head
        Input:  x,  x.shape  -->  (b, d, h, w).
        Output: y,  y.shape  -->  (b, d, l).
    """ 
    def __init__(
        self, 
        dim_sub,
        d_state,        
        scheme=3,
        gfu_t=0.0,
        pattern='C',
        route_dict=None,
        **kwargs,
    ):
        super().__init__()
       
        self.scheme = scheme      
        self.scan_pattern = pattern
        self.route_dict = route_dict 
        
        self.esfusion = GFU(t=gfu_t)       
        self.mambaeng = MambaEngine(d_model=dim_sub, d_state=d_state, **kwargs)
        self.scanning = ScanPattern(pattern=self.scan_pattern, s2s_engine=self.mambaeng, route_dict=self.route_dict)
        
        
    def forward(self, x: torch.Tensor):
        
        ys    = self.scanning(x)
        y_esf = self.esfusion(ys)
        
        return y_esf


# Multi-Head Scan Module    
class MHSM(nn.Module):
    r""" 
        Input:  x,  y.shape  -->  (b, h, w, c).
        Output: y,  y.shape  -->  (b, h, w, c).
    """     
    def __init__(
        self,
        d_model,
        d_state: int = 16,
        device = None,
        dtype = None,
        para_dict = {'head_num': 3, 'gfu_t': 0, 'with_proj': True}, 
        route_dict = None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model    = d_model         
        self.head_num   = para_dict['head_num']
        self.dim_sub    = self.d_model // self.head_num
        self.x_ssproj   = nn.Linear(self.d_model, self.dim_sub*self.head_num, bias=False)
        self.x_ssproj_W = self.x_ssproj.weight 
            
        self.scanhead_C = ScanHead(dim_sub=self.dim_sub, d_state=d_state, gfu_t=para_dict['gfu_t'], pattern='C', route_dict=route_dict)
        self.scanhead_X = ScanHead(dim_sub=self.dim_sub, d_state=d_state, gfu_t=para_dict['gfu_t'], pattern='X', route_dict=route_dict)
        self.scanhead_S = ScanHead(dim_sub=self.dim_sub, d_state=d_state, gfu_t=para_dict['gfu_t'], pattern='S', route_dict=route_dict)
        if self.head_num == 4:
            self.scanhead_Z = ScanHead(dim_sub=self.dim_sub, d_state=d_state, gfu_t=para_dict['gfu_t'], pattern='Z', route_dict=route_dict)
            
        self.with_proj = para_dict['with_proj']

        if self.with_proj:
            self.cat_norm = nn.LayerNorm(self.head_num*self.dim_sub)
            self.y_proj = nn.Linear(self.head_num*self.dim_sub, self.d_model, bias=False)
        
        
    def forward(self, x: torch.Tensor):
        
        b, d, h, w = x.shape
        
        if self.head_num == 3:
            x_CXS = torch.einsum("b d h w, c d -> b c h w", x, self.x_ssproj_W)
            ### x_C, x_X, x_S = rearrange(x_CXS, 'b (k d) h w -> k b d h w', k=n)
            x_C, x_X, x_S = torch.split(x_CXS, split_size_or_sections=self.dim_sub, dim=1)

            ## scan in subspaces
            y_C = self.scanhead_C(x_C)
            y_X = self.scanhead_X(x_X)
            y_S = self.scanhead_S(x_S)
        
            ## concatenate, layernorm and project
            y_cat = torch.cat([y_C, y_X, y_S], dim=1)            
            
        elif self.head_num == 4:
            x_CXSZ = torch.einsum("b d h w, c d -> b c h w", x, self.x_ssproj_W)
            ### x_Z, x_C, x_X, x_S = rearrange(x_ZCXS, 'b (k d) h w -> k b d h w', k=n)
            x_C, x_X, x_S, x_Z = torch.split(x_CXSZ, split_size_or_sections=self.dim_sub, dim=1)

            ## scan in subspaces
            y_C = self.scanhead_C(x_C)
            y_X = self.scanhead_X(x_X)
            y_S = self.scanhead_S(x_S)
            y_Z = self.scanhead_Z(x_Z)
            ## concatenate, layernorm and project
            y_cat = torch.cat([y_C, y_X, y_S, y_Z], dim=1)  
        
        if self.with_proj:
            y_ = self.y_proj(self.cat_norm(y_cat.transpose(-1,-2)).contiguous()) # b, l, d
        else:
            y_ = y_cat.transpose(-1,-2).contiguous() # b, l, d
            
        y = y_.view(b, h, w, -1)
        
        return y # y.shape (b, h, w, d)


# Visual State Space without Gate    
class VSSUnGated(nn.Module):
    r""" 
        Input:  x,  y.shape  -->  (b, h, w, d).
        Output: y,  y.shape  -->  (b, h, w, d).
    """      
    def __init__(
        self,
        d_model,
        bias = False,
        d_conv = 3,
        conv_bias = True,
        d_state: int = 16,
        device = None,
        dtype = None,
        dropout = 0,
        para_dict={'head_num': 3, 'gfu_t': 0, 'with_proj': True},  
        route_dict = None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.d_model = d_model 
        self.d_inner = d_model
        
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

        self.mhs = MHSM(d_model=d_model, d_state=d_state, para_dict=para_dict, route_dict=route_dict, **kwargs)
        
        self.out_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None     
        
    def forward(self, x):    
        
        b, h, w, c = x.shape

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        
        y = self.mhs(x)
        
        y = self.out_norm(y)
        out = y

        if self.dropout is not None:
            out = self.dropout(out)
        return out

    
# Visual State Space with Gate    
class VSSGate(nn.Module):
    r""" 
        Input:  x,  y.shape  -->  (b, h, w, d).
        Output: y,  y.shape  -->  (b, h, w, d).
    """      
    def __init__(
        self,
        d_model,
        bias = False,
        d_conv = 3,
        conv_bias = True,
        d_state: int = 16,
        device = None,
        dtype = None,
        dropout = 0,
        para_dict = {'head_num': 3, 'gfu_t': 0, 'with_proj': True},  
        route_dict = None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
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

        self.mhs = MHSM(d_model=d_model, d_state=d_state, para_dict=para_dict, route_dict=route_dict, **kwargs)
        
        self.out_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
              
    def forward(self, x):    
        
        b, h, w, c = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        
        y = self.mhs(x)
        
        y = self.out_norm(y)
        out = y * F.silu(z)

        if self.dropout is not None:
            out = self.dropout(out)
        return out   