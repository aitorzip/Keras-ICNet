# Keras-ICNet
### [[paper]](https://arxiv.org/abs/1704.08545)

Keras implementation of Real-Time Semantic Segmentation on High-Resolution Images

## Requisites
- Python 3
- Keras 2.0 with Tensorflow backend
- A dataset, such as Cityscapes or Mapillary ([Mapillary](https://research.mapillary.com/) was used in this case). The dataset should be placed under a datasets directory, following the directory structure:

    .
    ├── ...
    ├── mapillary               
    │   ├── training
    │   │   ├── images        # Contains the input images
    │   │   └── instances     # Contains the target labels
    │   ├── validation
    │   │   ├── images        # Contains the input images
    │   │   └── instances     # Contains the target labels
    │   └── testing
    │   │   └── images        # Contains the input images
    └── ...


## Train
Issue ```./train --help``` for options to start a training session, default arguments should work out-of-the-box.

These are the results of training for 300 epochs ```./train --epochs 300```

### Training
![conv6_cls_categorical_accuracy](https://raw.githubusercontent.com/ai-tor/Keras-ICNet/master/output/conv6_cls_categorical_accuracy.png)
![conv6_cls_loss](https://raw.githubusercontent.com/ai-tor/Keras-ICNet/master/output/conv6_cls_loss.png)
![loss](https://raw.githubusercontent.com/ai-tor/Keras-ICNet/master/output/loss.png)

### Validation
![val_conv6_cls_categorical_accuracy](https://raw.githubusercontent.com/ai-tor/Keras-ICNet/master/output/val_conv6_cls_categorical_accuracy.png)
![val_conv6_cls_loss](https://raw.githubusercontent.com/ai-tor/Keras-ICNet/master/output/val_conv6_cls_loss.png)
![val_loss](https://raw.githubusercontent.com/ai-tor/Keras-ICNet/master/output/val_loss.png)

## Test
Issue ```./test --help``` for options to start a testing session, default arguments should work out-of-the-box.

### Output examples
![10](https://raw.githubusercontent.com/ai-tor/Keras-ICNet/master/output/10.png)
![07](https://raw.githubusercontent.com/ai-tor/Keras-ICNet/master/output/7.png)

## TODO
* Increase test accuracy (perform training with data augmentation, perform class weighting)
