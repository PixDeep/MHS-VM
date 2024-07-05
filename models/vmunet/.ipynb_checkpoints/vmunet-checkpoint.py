from .vmamba import VSSM

import torch
from torch import nn


class VMUNet(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1,
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 para_dict={'head_num': 3, 'gfu_t': 0, 'with_proj': True}, 
                 route_dict_path=None
                ):
        super().__init__()

        self.num_classes = num_classes

        # load scan route dict
        print('#----------Loading Scanroutes----------#')
        route_dict = torch.load(route_dict_path)
        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                           para_dict=para_dict,
                           route_dict=route_dict
                        )
    
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.vmunet(x)
        if self.num_classes == 1: return torch.sigmoid(logits)
        else: return logits
