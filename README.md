# Tomatoe-Healthy-Checker
A small artificial intelligence application for check if a tomatoe leaf is healthy!


I just had found a tomatoe dataset on kaggle with a total of 10.000 images of different diseases and healthy plants. In this repo I did not upload them because of space elimination and things like them. That's why it exits some .tensor files which contain the images as tensors. This decreases the memory sice. Also I did not uplaod the validation images. But you can simple copy them in the specific folder and run a test.

# Sample information
Conv-Layers: 4
Drop Out-Layer: Yes, one
Fully-Connected-Layers: 2 Linear
Train Time: ~5 days on my old laptop without CUDA
Correct prediction: 87.5%

# How to use
Download or clone repo and run "python main.py [cmd]". For training you have to add -train [epoch_amounts]. To test some images, enter -test [tests]
