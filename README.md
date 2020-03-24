# Deep H3 Loop Prediction
A deep residual network architecture to predict probability distributions of 
inter-residue distances and angles for CDR H3 loops in antibodies. This work is protected by https://creativecommons.org/licenses/by-nc/3.0/. Please cite:

* Ruffolo JA, Guerra C, Mahajan SP, Sulam J, & Gray JJ, "Geometric Potentials from Deep Learning Improve Prediction of CDR H3 Loop Structures," *bioRXiv* 2020. [doi:10.1101/2020.02.09.940254](https://doi.org/10.1101/2020.02.09.940254)

ResNet part of the code is re-implemented from https://github.com/KaimingHe/resnet-1k-layers which was based on \
https://github.com/facebook/fb.resnet.torch. Network architecture is based on that of Wang et al. ([RaptorX-Contact](https://github.com/j3xugit/RaptorX-Contact)), and geometric descriptors based on Yang et al. ([trRosetta](https://github.com/gjoni/trRosetta)) (references below).

## Trained Model 
Model trained on ~ 1400 antibodies from the SAbDab Database is available in
deeph3/models/

## Requirements and Setup
torch, tensorboard (2.1 or higher), biopython (see requirements.txt for the complete list)

Be sure that your PYTHONPATH environment variable has the deepH3-distances-orientations/ directory. On linux, use the
following command:
```
export PYTHONPATH="$PYTHONPATH:/absolute/path/to/deepH3-distances-orientations"
```

## Prediction
To predict the binned distance and angle matrices for a given antibody sequence (in a fasta file), run:
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

Other arguments can be listed using the `--help` or `-h` option.

## Training
To train a model using a non-redundant set of bound and unbound antibodies 
downloaded from SAbDab, run:
```
python train.py 
```

By default, structures are selected from SAbDab with paired VH/VL chains, a resolution of 3 A or better, and at most 99% sequence identity (ie, the set used in our [original preprint](https://doi.org/10.1101/2020.02.09.940254).)

Other arguments can be listed using the `--help` or `-h` option.

## Authors
* **Carlos Guerra** - [cguerramain](https://github.com/cguerramain)
* **Sai Pooja Mahajan** - [heiidii](https://github.com/heiidii)
* **Jeff Ruffolo** - [jeffreyruffolo](https://github.com/jeffreyruffolo)

## Research Advisors
* **Jeremias Sulam** - [jsulam](https://github.com/jsulam)
* **Jeffrey Gray** - [jjgray](https://github.com/jjgray)

## References
1. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Identity Mappings in Deep 
   Residual Networks," *ECCV*, 2016.
   [arXiv:1603.05027](https://arxiv.org/abs/1603.05027)
2. S. Wang, S. Sun, Z. Li, R. Zhang and J. Xu, "Accurate De Novo Prediction of 
   Protein Contact Map by Ultra-Deep Learning Model", *PLOS Computational 
   Biology*, vol. 13, no. 1, p. e1005324, 2017. Available:
   [10.1371/journal.pcbi.1005324.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005324)
3. K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” 
   2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
   Available: [arXix:1512.03385](https://arxiv.org/abs/1512.03385)
4. J. Yang, I. Anishchenko, H. Park, Z. Peng, S. Ovchinnikov and D. Baker, 
   “Improved protein structure prediction using predicted interresidue orientations.,” 
   Proceedings of the National Academy of Sciences, 2020. 
   [PNAS](https://www.pnas.org/content/117/3/1496.short)
5. B. D. Weitzner, D. Kuroda, N. Marze, J. Xu and J. J. Gray, “Blind prediction 
   performance of RosettaAntibody 3.0: grafting, relaxation, kinematic loop modeling, 
   and full CDR optimization.,” Proteins: Structure, Function, and Bioinformatics, 
   vol. 82, no. 8, pp. 1611–1623, 2014.
   [Wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/prot.24534)

