# Deep H3 Loop Prediction
A deep residual network architecture to predict probability distributions of 
inter-residue distances for CDR H3 loops in antibodies. This work is protected by https://creativecommons.org/licenses/by-nc/3.0/.

ResNet part of the code is re-implemented from https://github.com/KaimingHe/resnet-1k-layers which was based on \
https://github.com/facebook/fb.resnet.torch

## CLI Experimentation
This repository supports experimenting on H5 datasets for antibody data through a CLI.

## Model Training
Here is an example of a command to train a model on antibody H5 datasets 
located in the /foo/bar/antibody_h5 directory. The working directory is 
deeph3 and the model is saved every 20 batches to the 
/baz/boo/models directory.
```buildoutcfg
python ./cli/antibody_trainer.py /foo/bar/antibody_h5 \
--validation_file antibody_validation.h5 --training_file antibody_validation.h5 \
--epochs 4 --batch_size 1 --save_every 20 --models_dir /baz/boo/models
```

## Generating H5 file from pdb and fasta files
To generate H5 files from pdb and fastas files, use the command below from the
deeph3 working directory with pdb files stored in /foo/bar/antibody_pdbs and 
fasta files stored in /foo/bar/antibody_fastas. The output h5 file in this example
is /foo/bar/antibody_h5/antibody_validation.h5.
```
python ./cli/generate_h5_antibody_files_cli.py /foo/bar/antibody_pdbs \
/foo/bar/antibody_fastas /foo/bar/antibody_h5/antibody_validation.h5
```
For example pdb and fasta files see: /deeph3/data/antibody_dataset/fastas_testrun/
and /deeph3/data/antibody_dataset/pdbs_testrun/

Use the '-h' or '--help' flag for a full list and description of flags.

## Trained Model 
Model trained on ~ 1400 antibodies from the SABDAb Database is available here
/deeph3/models/adam_opt_lr01_da/

## Requirements

torch, tensorboard (1.4 or higher), biopython (see requirements.txt for the complete list)

## Authors
* **Carlos Guerra** - [cguerramain](https://github.com/cguerramain)
* **Sai Pooja Mahajan** - [heiidii](https://github.com/heiidii)
* **Jeff Ruffolo** - [jeffreyruffolo])https://github.com/jeffreyruffolo)

## Research Advisors
* **Jeremias Sulam**
* **Jeffrey Gray** - [jjgray](https://github.com/jjgray)

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

