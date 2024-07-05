import os
from argparse import ArgumentParser

import torch
import numpy as np

from scanutils import *


def parse_args( ):
    parser = ArgumentParser( )
    parser.add_argument("--h", type=int, default=256)
    parser.add_argument("--w", type=int, default=256)
    parser.add_argument("--d", type=int, default=  4)
    return vars(parser.parse_args( ))


def calc_posarrays(H, W, Depth=4):
    
    scale = np.power(2, Depth+1) 
    if H % scale != 0 or W % scale != 0:
        print('Please ensure that {} divides both H and W exactly.'.format(scale))
        return
    
    H0, W0 = H//4, W//4
    dict_list = []
    print('The network hierarchy is as follows:')
    for i in np.arange(Depth):
        
        scale = np.power(2, i)    
        h, w = H0//scale, W0//scale
        print('|--- layer_{}: h, w = {} x {}'.format(i, h, w))
        
        # start scanning
        dict_pos = {}

        ## CScan
        cscan_0_pos2d = cons_scan_h(h, w)
        cscan_0_tind, cscan_0_pind = pos2d_indices(cscan_0_pos2d, w)
        cscan_0_tpind = torch.stack([cscan_0_tind, cscan_0_pind], dim=0)
        
        cscan_1_pos2d = cons_scan_v(h, w)
        cscan_1_tind, cscan_1_pind = pos2d_indices(cscan_1_pos2d, w)
        cscan_1_tpind = torch.stack([cscan_1_tind, cscan_1_pind], dim=0)

        ## XScan
        xscan_0_pos2d = diag_scan(h, w, cw=True)
        xscan_0_tind, xscan_0_pind = pos2d_indices(xscan_0_pos2d, w)
        xscan_0_tpind = torch.stack([xscan_0_tind, xscan_0_pind], dim=0)
        
        xscan_1_pos2d = xscan_0_pos2d.clone()
        xscan_1_pos2d[:,1] = w-1 - xscan_1_pos2d[:,1]
        xscan_1_tind, xscan_1_pind = pos2d_indices(xscan_1_pos2d, w)
        xscan_1_tpind = torch.stack([xscan_1_tind, xscan_1_pind], dim=0)
        
        ## SScan
        sscan_0_pos2d = spiral_scan(h, w)
        sscan_0_tind, sscan_0_pind = pos2d_indices(sscan_0_pos2d, w)
        sscan_0_tpind = torch.stack([sscan_0_tind, sscan_0_pind], dim=0)
        
        sscan_1_pos2d = sscan_0_pos2d.clone()
        sscan_1_pos2d[:,0] = h-1 - sscan_1_pos2d[:,0]
        sscan_1_tind, sscan_1_pind = pos2d_indices(sscan_1_pos2d, w)
        sscan_1_tpind = torch.stack([sscan_1_tind, sscan_1_pind], dim=0)
        
        ## ZScan
        zscan_0_pos2d, zscan_1_pos2d = zigzag_scan(h, w)
        zscan_0_tind, zscan_0_pind = pos2d_indices(zscan_0_pos2d, w)
        zscan_0_tpind = torch.stack([zscan_0_tind, zscan_0_pind], dim=0)
        
        zscan_1_tind, zscan_1_pind = pos2d_indices(zscan_1_pos2d, w)
        zscan_1_tpind = torch.stack([zscan_1_tind, zscan_1_pind], dim=0)
        
        
        cxs_tpinds = torch.stack([cscan_0_tpind, cscan_1_tpind, xscan_0_tpind, xscan_1_tpind, 
                                  sscan_0_tpind, sscan_1_tpind, zscan_0_tpind, zscan_1_tpind], dim=0)
        
        dict_pos['layer_'+str(i)+'.cxsz_tpinds'] = cxs_tpinds
        
        # append
        dict_list.append(dict_pos)
        
    return dict_list



def main(h, w, d=4):
    print('H = {}, W = {}, Depth = {}'.format(h, w, d))
    route_dict = calc_posarrays(H=h, W=w, Depth=d)
    filename = './routedicts/route_dict_{}_{}_{}l.pth'.format(h, w, d)
    torch.save(route_dict, filename)
   
   
if __name__ == "__main__":
    main(**parse_args())