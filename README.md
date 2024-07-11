This is the official code repository for [MHS-VM](https://arxiv.org/pdf/2406.05992).

<div align="center">
<h1> MHS-VM </h1>
<h3> Multi-Head Scanning in Parallel Subspaces for Vision Mamba </h3>
(https://arxiv.org/pdf/2406.05992)
</div>

### Release

- **News**: `2024/07/05`: MHS Module and MHS-VM released.

### Introduction

A Multi-Head Scan (MHS) mechanism is introduced to enhance visual representation learning.


![module](https://github.com/PixDeep/MHS-VM/blob/main/assets/Figure-1.png)

A richer array of scan patterns is introduced to capture the diverse visual patterns present in vision data.

![Scan Patterns](https://github.com/PixDeep/MHS-VM/blob/main/assets/Figure-2.png)

A Scan Route Attention (SRA) mechanism is introduced to enable the model to attenuate or screen out trivial
features, thereby enhancing its ability to capture complex structures in images

![Embedding Section Fusion](https://github.com/PixDeep/MHS-VM/blob/main/assets/Figure-5.png)

In our experiments, we examine the CV for the relative deviations of the $k$ values, providing insights into the variability and consistency of the embeddings' responses along different scan routes. We facilitate the module's ability to selectively filter or attenuate information through the incorporation of a multiplicative gating mechanism based on the relative CV. This process is formulated as:

$$
\begin{equation}
z = (\sum_{i=1}^{k} y_i) \odot \sigma(y_{cv})
\end{equation}
$$

where $y_{cv} = \text{std}([y_i]) / \text{avg}([y_i-\text{min}([y_i])])$ represents the relative CV, and $\odot$ denotes the element-wise product between tensors, and $\sigma(x)$ is a monotone function, such as Sigmoid, ReLU, power function and exponential function $\exp(\cdot)$, etc. This monotone function is introduced to prompt the Mamba block to extract position-aware features. 

$$
\begin{equation}
\sigma(x, t) = \text{ReLU}(x-t) = \text{max}(0, x-t)
\end{equation}
$$

This function returns $0$ when $x < t$ and $x-t$ otherwise. The parameter $t$ can be set as a hyperparameter or a learnable parameter. Such a strategy can be considered as a novel regularization technique to prevent over-fitting and improve generalization. 

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

## Test

```python
cd MHS-VM
python test.py --h 4 --d isic2018 --p best_4h.pth
```

```python
miou: 0.8085252327081669, f1_or_dsc: 0.8941265712919525
```

An interesting observation is that the model, which was trained using the dataset isic2018, might yield notably high performance when evaluated against the test set of the dataset isic2017.

```python
cd MHS-VM
python test.py --h 4 --d isic2017 --p best_4h.pth
```

```python
miou: 0.8201665691022297, f1_or_dsc: 0.9011994649553033
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
