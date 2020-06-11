import cv2
import argparse, time, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as Variable
import torch.optim as optim
from classifier import TomatoNetwork, train_tomatoes, test_tomatoes, make_imgs_tensors
import config

parser = argparse.ArgumentParser('python[3] main.py')
parser.add_argument('-train', '--train', help='start a training for some epochs. Epochrange has to be given or -1 for an endless loop. To stop press Crtl-C')
parser.add_argument('-test', '--test', help='test some images and given feedback. Count of images has to be given or -1 for all')
parser.add_argument('-newtensors', '--newtensors', help='recalculate all images to tensors!')
args = parser.parse_args()


def main():  
    start = time.time() 

    tomato_network = TomatoNetwork()
    if os.path.isfile(config.TOMATO_NETWORK_FILE_NAME):
        tomato_network = torch.load(config.TOMATO_NETWORK_FILE_NAME)

    if args.train:
        tomato_network = train_tomatoes(tomato_network, epochs=int(args.train), batches_loaded_count=50)
    if args.newtensors:
        make_imgs_tensors()
    if args.test:
        tomato_network = test_tomatoes(tomato_network, tests=int(args.test))

    print(f'Program take {time.time() - start} seconds to run!')

if __name__ == "__main__":
    main()