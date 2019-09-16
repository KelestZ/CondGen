# GVGANs
This is a pytorch implementation of our paper:
[Conditional Structure Generation through Graph Variational Generative Adversarial Nets](https://nips.cc/Conferences/2019/AcceptedPapersInitial), Neural Information Processing Systems (NeurIPS'19)

Carl Yang, Peiye Zhuang, Wenhan Shi, Alan Luu, Pan Li

## Prerequisites
- Python3
- Pytorch 0.4
- Tookits like python-igraph, powerlaw, networkx etc.

## Data
We release our [DBLP dataset](https://drive.google.com/open?id=1s9hLOEAIL4j63fBpIdm1IldfJCsLhzpB) and [TCGA dataset](https://drive.google.com/open?id=1s9hLOEAIL4j63fBpIdm1IldfJCsLhzpB) on Google Drive.

## Training 
```
python train.py
```
with default setttings in `options.py`.


## Citation

If you use this code for your research, please cite our paper:
```
@article{gvgans,
  title={Conditional Structure Generation through Graph Variational Generative Adversarial Nets},
  author={Carl Yang, Peiye Zhuang, Wenhan Shi, Alan Luu, Pan Li},
  journal={Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```
