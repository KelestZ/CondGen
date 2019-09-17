## Implementation of *CondGen*, NeurIPS 2019.

Please cite the following work if you find the code useful.

```
@inproceedings{yang2018meta,
	Author = {Yang, Carl and Zhuang, Peiye and Shi, Wenhan and Luu, Alan and Pan, Li},
	Booktitle = {NeurIPS},
	Title = {Conditional structure generation through graph variational generative adversarial nets},
	Year = {2019}
}
```
Contact: Peiye Zhuang (peiye@illinois.edu), Carl Yang (yangji9181@gmail.com)


![Results](https://github.com/KelestZ/GVGAN/blob/master/misc/gvgan-new1.png)


## Prerequisites
- Python3
- Pytorch 0.4
- Tookits like python-igraph, powerlaw, networkx etc.

## Data
Our [DBLP dataset](https://drive.google.com/open?id=1s9hLOEAIL4j63fBpIdm1IldfJCsLhzpB) and [TCGA dataset](https://drive.google.com/open?id=1voorUDkxJc0kzA7AEDCf1Pk4opjCGvXL) are released on Google Drive.

## Training 
```
python train.py
```
with default setttings in `options.py`.

