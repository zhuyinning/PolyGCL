## PolyGCL: GRAPH CONTRASTIVE LEARNING via Learnable Spectral Polynomial Filters

This repository contains a PyTorch implementation of ICLR 2024 paper "[*PolyGCL: GRAPH CONTRASTIVE LEARNING via Learnable Spectral Polynomial Filters*](https://openreview.net/pdf?id=y21ZO6M86t)".


## Dependencies
Please setup the environment with Python 3.10 and CUDA 11.8. Typically, you might need to run the following commands:

```
pip install --upgrade pip setuptools wheel
pip install numpy==1.26.4
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 \
--index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
-f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install networkx matplotlib seaborn==0.13.2 alive-progress
```

### Datasets
We provide the small datasets in the folder 'data'. You can access the heterophilic datasets and the large heterophilic graph arXiv-year via [heterophilous-graphs](https://github.com/yandex-research/heterophilous-graphs) and [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale) respectively.


## Reproduce the results

### On real-world datasets
You can run the following commands directly.

```sh
sh exp_PolyGCL.sh
```
Heterophilic datasets:
```sh
cd HeterophilousGraph
sh exp_PolyGCL.sh
```
Large heterophilic graph arXiv-year:
```sh
cd non-homophilous
sh exp_PolyGCL.sh
```

### On synthetic datasets

Generate the cSBM data firstly.
```sh
cd cSBM
sh create_cSBM.sh
```
Then run the following command directly.
```sh
sh run_cSBM.sh
```

## Acknowledgements 
This project includes code or ideas inspired by the following repositories:
 - [ChebNetII](https://github.com/ivam-he/ChebNetII)
 -  [MVGRL](https://github.com/kavehhassani/mvgrl)
 - [DGI](https://github.com/PetarV-/DGI)
 
## Citation

    @inproceedings{
	    chen2024polygcl,
	    title={Poly{GCL}: {GRAPH} {CONTRASTIVE} {LEARNING} via Learnable Spectral Polynomial Filters},
	    author={Jingyu Chen and Runlin Lei and Zhewei Wei},
	    booktitle={The Twelfth International Conference on Learning Representations},
	    year={2024},
	    url={https://openreview.net/forum?id=y21ZO6M86t}
    }

## Contact
If you have any questions, please feel free to contact me with [jy.chen@ruc.edu.cn](mailto:jy.chen@ruc.edu.cn).
