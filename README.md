# Deep H3 Loop Prediction
A deep residual network architecture to predict probability distributions of 
inter-residue distances and angles for CDR H3 loops in antibodies. This work is protected by https://creativecommons.org/licenses/by-nc/3.0/.

ResNet part of the code is re-implemented from https://github.com/KaimingHe/resnet-1k-layers which was based on \
https://github.com/facebook/fb.resnet.torch

Details of implementation, ResNet architecture and datasets for training and validation are available
here:
https://www.biorxiv.org/content/10.1101/2020.02.09.940254v1

## Trained Model 
Model trained on ~ 1400 antibodies from the SABDAb Database is available here
deeph3/models/

## Training dataset 
We used thresholds of 99% sequence identity and 3.0 Å resolution to obtain the training dataset from the non-redundant database on SABDAb(http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/search/).

## Usage
```
python predict.py --fasta_file [fasta file path] --model [model file path]
```
The fasta file must have the following format:
```
>[PDB ID]:H	[heavy chain sequence length]
[heavy chain sequence]
>[PDB ID]:L	[light chain sequence length]
[light chain sequence]
```
See deeph3/data/antibody_dataset/fastas_testrun for an example.

Other optional arguments can be listed using the --help or -h option.

## Requirements
torch, tensorboard (1.4 or higher), biopython (see requirements.txt for the complete list)

## Authors
* **Carlos Guerra** - [cguerramain](https://github.com/cguerramain)
* **Sai Pooja Mahajan** - [heiidii](https://github.com/heiidii)
* **Jeff Ruffolo** - [jeffreyruffolo](https://github.com/jeffreyruffolo)

## Research Advisors
* **Jeremias Sulam** - [jsulam](https://github.com/jsulam)
* **Jeffrey Gray** - [jjgray](https://github.com/jjgray)

## Cite
If you publish or report results using this work, please cite:
Jeffrey A. Ruffolo, Carlos Guerra, Sai Pooja Mahajan, Jeremias Sulam, Jeffrey J. Gray
bioRxiv 2020.02.09.940254; doi: https://doi.org/10.1101/2020.02.09.940254

## References
1. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Identity Mappings in Deep 
   Residual Networks. In *ECCV*, 2016.
   [arXiv:1603.05027](https://arxiv.org/abs/1603.05027)
2. S. Wang, S. Sun, Z. Li, R. Zhang and J. Xu, "Accurate De Novo Prediction of 
   Protein Contact Map by Ultra-Deep Learning Model", *PLOS Computational 
   Biology*, vol. 13, no. 1, p. e1005324, 2017. Available:
   [10.1371/journal.pcbi.1005324.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005324)
3. K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” 
   2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
   Available: [arXix:1512.03385](https://arxiv.org/abs/1512.03385)

