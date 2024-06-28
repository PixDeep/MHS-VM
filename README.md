This is the official code repository for [MHS-VM](https://arxiv.org/pdf/2406.05992).

<div align="center">
<h1> [MHS-VM] </h1>
<h3> Multi-Head Scanning in Parallel Subspaces for Vision Mamba </h3>
(https://arxiv.org/pdf/2406.05992)
</div>

### Release

- **News**: `2024/6/30`: MHS Module and MHS-VM released.

![module](https://github.com/PixDeep/MHS-VM/blob/main/assets/Figure-1.png)

![Embedding Section Fusion](https://github.com/PixDeep/MHS-VM/blob/main/assets/Figure-5.png)

![Scan Patterns](https://github.com/PixDeep/MHS-VM/blob/main/assets/Figure-2.png)

## Main Environments

The environment installation can follow the work [VM-UNet](https://github.com/JCruan519/VM-UNet), or the steps below:

```shell
conda create -n mhsvm python=3.10
conda activate mhsvm
pip install torch==2.0.1 torchvision==0.15.2
pip install packaging==24.0
pip install timm==1.0.3
pip install triton==2.0.0
pip install causal_conv1d==1.2.0 
pip install mamba_ssm==1.2.0
pip install tensorboardX  
pip install pytest chardet yacs termcolor
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```

## Datesets

For datasets, please refer to [VM-UNet](https://github.com/JCruan519/VM-UNet) for further details.

## Scan Route Dictionary

Since the scan routes are fixed within the model, we opt to pre-generate the route hierarchy and store it in a dictionary. To accommodate various resolutions, you can generate the scan routes using the following command:

```python
python routegen.py --w 512 --h 512
```

## Train

```python
cd MHS-VM
python train.py
```

## Citation

If you find this repository useful, please cite our work: 

```
@misc{ji2024mhsvmmultiheadscanningparallel,
      title={MHS-VM: Multi-Head Scanning in Parallel Subspaces for Vision Mamba}, 
      author={Zhongping Ji},
      year={2024},
      eprint={2406.05992},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2406.05992}, 
}
```

## Acknowledgments

This code is based the [VM-UNet](https://arxiv.org/abs/2402.02491).
