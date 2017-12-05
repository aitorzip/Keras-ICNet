# Keras-ICNet
### [[paper]](https://arxiv.org/abs/1704.08545)

Keras implementation of Real-Time Semantic Segmentation on High-Resolution Images. **Training in progress!**
    
## Requisites
- Python 3.6.3
- Keras 2.1.1 with Tensorflow backend
- A dataset, such as Cityscapes or Mapillary ([Mapillary](https://research.mapillary.com/) was used in this case).

## Train
Issue ```./train --help``` for options to start a training session, default arguments should work out-of-the-box.

You need to place the dataset following the next directory convention:

    .
    ├── mapillary                   
    |   ├── training
    |   |   ├── images             # Contains the input images
    |   |   └── instances          # Contains the target labels
    |   ├── validation
    |   |   ├── images
    |   |   └── instances
    |   └── testing
    |   |   └── images
    

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
* Perform class weighting
