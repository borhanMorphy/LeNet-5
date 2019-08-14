## TITLE 
Tensorflow implementation of LeNet-5 architecture

## MODEL
![Model](LeNet-5.jpg)

## HOW TO USE
**Arguments**:
- mode              (must be selected between these values:"train","test","deploy")
- cores             (specifies how many cores will be used, default is 1)
- load_model        (path for pre-trained model)

**Train Mode Only Arguments**:
- save_top          (specifies how many models will be selected to save with respect to cost value while training, default is 3)
- batch_size        (batch size for training, default is 128)
- epoch             (epoch count for training, default is 10)
- learning_rate     (learning rate for training, default is 0.001)

### Training

example:
python model.py --mode train --learning_rate 0.002 --epoch 15 --cores 4 --batch_size 512 --save_top 5

## REFERENCE
[Related Paper Link](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
