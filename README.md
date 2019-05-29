# wide-resnet-pytorch

## requirements 

・python3.7 

・cudax.x 

・pytorch 

`conda install pytorch torchvision cudatoolkit=x.x -c pytorch` 

## performance

CIFAR100

・TestAcc 81.60%(max accuracy over 4 runs)

## implementation
・When using default hyperparameters configure

`python train.py`

・When trying some other hyperparameters

e.g. without dropout

`python train.py -drop_rates1=0.0 -drop_rates2=0.0 -drop_rates3=0.0`
