# Vertebra Segmentation

This repository contains the code for semantic segmentation of vertebra using the PyTorch framework.

Models:
- Unet
- Unet++
- Att-Unet
- SalsaNext
- Transformer

# How to use
1. Change data path in train.py (line 58-63) and test.py (line 50-51)
2. You can modify the type of model you want to train in train.py line 96, options:
- model = SalsaNext()
- model = AttU_Net()
- model = NestedUNet()
- model = build_unet() #original Unet
- model = U_Transformer()

3. Run python train.py to train model
