import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.vmunet.vmunet import VMUNet

from engine import *
import os
import sys
from argparse import ArgumentParser
from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")


def parse_args( ):
    parser = ArgumentParser( )
    parser.add_argument("--h", type=int, default=4)
    parser.add_argument("--d", type=str, default='isic2018')
    parser.add_argument("--p", type=str, default='best_4h.pth')

    return vars(parser.parse_args( ))


def main(config, h=4, d='isic2018', p='best_4h.pth'):

    checkpoint_dir = 'pretrained/'
    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    datapath = os.path.join('../VM-UNet/data', d+'/')
    val_dataset = NPY_datasets(datapath, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    model_cfg['para_dict']['head_num'] = h
    model_cfg['para_dict']['gfu_t'] = 0.4
    print('model_cfg: ', model_cfg)
    model = VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        para_dict=model_cfg['para_dict'],
        route_dict_path=model_cfg['route_dict_path']
    )
    model = model.cuda()
    model.eval()
    input = torch.randn(1, 3, 256, 256).cuda()
    flops, params = profile(model, inputs=(input,))
    criterion = config.criterion
    
    resume_model = os.path.join(checkpoint_dir, p)
    best = torch.load(resume_model, map_location=torch.device('cpu'))
    model.load_state_dict(best)
    
    print('#----------Testing Model----------#')
    log_dir = os.path.join(config.work_dir, 'log')
    logger = get_logger('train', log_dir)
    test_one_epoch(
        val_loader,
        model,
        criterion,
        logger,
        config,
    )

  
if __name__ == "__main__":
    
    config = setting_config
    main(config, **parse_args())
