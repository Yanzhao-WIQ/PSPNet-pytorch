# PSPNet-pytorch
PyTorch implementation of PSPNet segmentation network


### Original paper

 [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)
 
### Details

* The basebone of pspnet is *ResNet-50*
* The code has been tested on Python 3.8 (torch,albumentations,numpy, and matplotlib are needed )
* The model was trained on cpu,gpu may also be available.
* The file folder and file description  is as follows:

````
├────data├──── images(contain 328 Weizmann Horse images)
│        ├──── masks(contain 328 Weizmann Horse masks)
│        ├──── val_result(contain test results)
│
├────model_saved(contain trained models)
│   
├────dataset.py(set data)     
├────Model.py(pspnet model)    
├────train.py(train)     
├────test.py(test)     
````
* The evaluation metrics of model is MIoU, but may have a little problem.


