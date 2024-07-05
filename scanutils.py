import torch
from einops import rearrange


# cw: clockwise; ccw: anticlockwise 
def diag_scan(h, w, cw=True):
    
    pos2d = torch.LongTensor([])
    i = torch.arange(h, dtype=torch.long).unsqueeze(-1).expand(-1,  w)
    j = torch.arange(w, dtype=torch.long).unsqueeze( 0).expand( h, -1)
    
    for k in torch.arange(h+w-1):
        mask = i+j == k
        pos_ij = torch.stack([i[mask], j[mask]], dim=-1)
        
        if (k+cw)%2 == 1:
            pos_ij = pos_ij.flip(dims=[0])
            
        pos2d = torch.cat((pos2d, pos_ij))
        
    return pos2d


# Consecutively scan.
## Horizontally
def cons_scan_h(h, w):
    pos_i = torch.empty(0, dtype=torch.long)
    pos_j = torch.empty(0, dtype=torch.long)

    for k in torch.arange(h):
        pos_i_ = torch.full((w,), k, dtype=torch.long)
        pos_j_ = torch.arange(0, w, dtype=torch.long)
        if k%2 == 1:
            pos_j_ = torch.flip(pos_j_, dims=[-1])
        # cat
        pos_i = torch.cat([pos_i, pos_i_])
        pos_j = torch.cat([pos_j, pos_j_]) 

    pos2d = torch.stack([pos_i, pos_j], dim=-1)
    return pos2d  


## Vertically
def cons_scan_v(h, w):
    pos_i = torch.empty(0, dtype=torch.long)
    pos_j = torch.empty(0, dtype=torch.long)

    for k in torch.arange(w):
        pos_i_ = torch.arange(0, h, dtype=torch.long)
        pos_j_ = torch.full((h,), k, dtype=torch.long)

        if k%2 == 1:
            pos_i_ = torch.flip(pos_i_, dims=[-1])
        # cat
        pos_i = torch.cat([pos_i, pos_i_])
        pos_j = torch.cat([pos_j, pos_j_]) 

    pos2d = torch.stack([pos_i, pos_j], dim=-1)
    return pos2d  


# clockwise spiral scan
def spiral_scan(h, w):
    ## no (h == 1 and w==1)
    pos_i = torch.empty(0, dtype=torch.long)
    pos_j = torch.empty(0, dtype=torch.long)
    
    # top left and bottom right
    tl_h, tl_w = 0, 0
    br_h, br_w = h-1, w-1 
    while True:
        # add a circle
        # i
        pos_i_ = torch.cat([torch.full((br_w-tl_w,), tl_h, dtype=torch.long),
                   torch.arange(tl_h, br_h, dtype=torch.long),
                   torch.full((br_w-tl_w,), br_h, dtype=torch.long),
                   torch.arange(br_h, tl_h, -1, dtype=torch.long)])
        # j
        pos_j_ = torch.cat([torch.arange(tl_w, br_w, dtype=torch.long),
                   torch.full((br_h-tl_h,), br_w, dtype=torch.long),
                   torch.arange(br_w, tl_w, -1, dtype=torch.long),
                   torch.full((br_h-tl_h,), tl_w, dtype=torch.long)])
        # cat
        pos_i = torch.cat([pos_i, pos_i_])
        pos_j = torch.cat([pos_j, pos_j_])     

        # update anchors
        tl_h, tl_w, br_h, br_w = tl_h+1, tl_w+1, br_h-1, br_w-1   
        
        if tl_h > br_h or tl_w > br_w:
            break
        elif tl_h == br_h and tl_w == br_w:
            pos_i = torch.cat([pos_i, torch.LongTensor([tl_h])])
            pos_j = torch.cat([pos_j, torch.LongTensor([tl_w])])

    pos2d = torch.stack([pos_i[:h*w], pos_j[:h*w]], dim=-1)
    return pos2d


# Z-shape scan
def zigzag_scan(h, w):

    i, j = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    
    i_h = i.flatten( )
    j_h = j.flatten( )
    pos2d_h = torch.stack((i_h, j_h), dim=-1)
   
    i_v = i.t().flatten( )
    j_v = j.t().flatten( )
    pos2d_v = torch.stack([i_v, j_v], dim=-1)
    
    return pos2d_h, pos2d_v


def pos2d_indices(pos_arr, w):

    tindex = pos_arr[:,0]*w+pos_arr[:,1]    
    pindex = torch.empty_like(tindex, dtype=torch.long)
    pindex[tindex] = torch.arange(pos_arr.shape[0], dtype=torch.long)

    return tindex, pindex


def index_select_2d(x, index):

    b, d, h, w = x.shape
    x_ = x.view(b, d, -1)
    out = torch.index_select(x_, dim=-1, index=index)
    
    return out

