# Targeted Attack of Deep Hashing via Prototype-supervised Adversarial Networks
This is the available code for our TMM paper [Targeted Attack of Deep Hashing via Prototype-supervised Adversarial Networks](https://ieeexplore.ieee.org/abstract/document/9488305/).

## Usage
#### Dependencies
- Python 3.7.6
- Pytorch 1.6.0
- Numpy 1.18.5
- Pillow 7.1.2
- CUDA 10.2


#### Train hashing models
Initialize the hyper-parameters in hashing.py following the paper, and then run
```
python hashing.py
```

#### Attack by P2P or DHTA
Initialize the hyper-parameters in dhta.py following the paper, and then run
```
python dhta.py
```

#### Train ProS-GAN
Initialize the hyper-parameters in main.py following the paper, and then run
```
python main.py --train True
```

#### Evaluate ProS-GAN
Initialize the hyper-parameters in main.py following the paper, and then run
```
python main.py --train False --test True
```

## Cite
If you find this work is useful, please cite the following:
```
@article{zhang2021targeted,
  title={Targeted Attack of Deep Hashing via Prototype-supervised Adversarial Networks},
  author={Zhang, Zheng and Wang, Xunguang and Lu, Guangming and Shen, Fumin and Zhu, Lei},
  journal={IEEE Transactions on Multimedia},
  year={2021},
  publisher={IEEE}
}
```