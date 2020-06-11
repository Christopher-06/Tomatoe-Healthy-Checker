import cv2
import argparse, time, os, sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from threading import Thread
import config

transform = transforms.Compose([
    transforms.ToTensor() #return (3, 256, 256)
    #.unsqueeze_(0) würde (1, 3 ,256 ,256) returnen, also eine Dimension hinzufügen
])

class TomatoNetwork(nn.Module):
    def __init__(self):
        super(TomatoNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 8, kernel_size=5)
        self.conv3 = nn.Conv2d(8, 10, kernel_size=5)
        self.conv4 = nn.Conv2d(10, 12, kernel_size=5)
        self.conv_drop = nn.Dropout2d(p=0.4)
        self.fc1 = nn.Linear(1728, 1000)
        self.fc2 = nn.Linear(1000,10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #conv layers
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.conv_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = x.view(-1, 1728)
        #Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.sigmoid(x)

def get_target(name : str):
    return [
        1 if 'Bacterial_spot' in name else 0,
        1 if 'Early_blight' in name else 0,
        1 if 'healthy' in name else 0,
        1 if 'Late_blight' in name else 0,
        1 if 'Leaf_Mold' in name else 0,
        1 if 'Septoria_leaf_spot' in name else 0,
        1 if 'Spider_mites' in name else 0,
        1 if 'Target_Spot' in name else 0,
        1 if 'Tomato_mosaic_virus' in name else 0,
        1 if 'Tomato_Yellow_Leaf' in name else 0,
    ]

def tensor_to_highest(tensor):
    highest_index = 0
    highest = -100000
    for index, t in enumerate(tensor):
        if t > highest:
            highest = t
            highest_index = index

    arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    arr[highest_index] = 1
    return arr

def img_processing(img):
    return transform(img)

def extract_train_images():
    #was used to put all images in only one folder
    #that was needet to choose random images better!
    dirs = os.listdir('data/training/')
    for dir_name in tqdm(dirs):
        for f in os.listdir('data/training/' + dir_name):
            #make new filename with unique number and label and file type
            new_file_name = ''
            while new_file_name == '' or os.path.isfile('data/training/' + new_file_name):
                new_file_name = str(random.randint(100000, 999999)) + '_' + dir_name + '_' + f.replace(' ', '_')
            
            #make the new file
            img = cv2.imread('data/training/' + dir_name + '/' + f)
            cv2.imwrite('data/training/' + new_file_name, img)
            os.remove('data/training/' + dir_name + '/' + f)

def get_total_batch_count():
    images = len(os.listdir('data/training/'))
    return int(images / config.TRAIN_BATCH_SIZE)

def get_train_batches(batch_count = -1):   
    print(f"Getting {batch_count} batches")
    batch_data = []
    train_data = []
    target_data = []

    #get all avaible image names
    images = os.listdir('data/training/')

    if batch_count <= 0:
        #if all batches should be returned, we set batch_count to the number
        #of images and then we are sure to get all
        #because the images are packed together by the Training_Batch_Size
        #So it gives less batches than images
        batch_count = len(images)

    for _ in tqdm(range(batch_count),desc='Prepare batches'):
        for _ in range(config.TRAIN_BATCH_SIZE):
            #make one batch ready
            if len(images) == 0:
                break
            
            #choose random
            f = random.choice(images)
            images.remove(f)

            #read image and make it to tensor
            img = cv2.imread('data/training/' + f)
            train_data.append(img_processing(img))
            target_data.append(get_target(f))

        #enter as tuple inside the batch_data array
        if len(train_data) > 0:
            batch_data.append((torch.stack(train_data), target_data))
            train_data = []
            target_data = []      

    print(f'Got {len(batch_data)}x{batch_data[0][0].size()}')
    return batch_data            

def make_imgs_tensors():
    #delete every .tensor file
    files = os.listdir('data/training/tensors')
    for f in files:
        os.remove('data/training/tensors/' + f)

    tensors = []
    files = os.listdir('data/training/')
    p_bar = tqdm(desc='Images proceed', total=len(files))
    while len(files) > 0:
        p_bar.update()
        f = random.choice(files)
        files.remove(f)

        if f == 'tensors':
            continue

        img = cv2.imread('data/training/' + f)
        tensors.append((img_processing(img), get_target(f)))

        if len(tensors) >= config.TENSORS_IN_FILES or len(files) == 0:
            new_file_name = ''
            while new_file_name == '' or os.path.isfile('data/training/tensors/' + new_file_name):
                new_file_name = str(random.randint(100000, 999999)) + '_Tensors.tensor'
            torch.save(tensors, 'data/training/tensors/' + new_file_name)
            tensors = []
    p_bar.close()

def get_batch():
    tensor_files = os.listdir('data/training/tensors')
    if len(tensor_files) == 0:
        print("No tensor_files avaible!")
        return []

    train_data, target_data = [], []
    while len(train_data) < config.TRAIN_BATCH_SIZE:
        f = random.choice(tensor_files)
        tensors = torch.load('data/training/tensors/' + f)

        for _ in range(0, random.randint(0, config.TRAIN_BATCH_SIZE - len(train_data))):
            choosen = random.choice(tensors)
            train_data.append(choosen[0])
            target_data.append(choosen[1])


    return [(torch.stack(train_data), target_data)]

class batch_loader():
    def __init__(self, batch_size : int, max_batches_loaded : int):
        self.batch_size = batch_size
        self.max_batches_loaded = max_batches_loaded
        self.batches = []
        self.stopping = False
        self.prepare_thread = Thread(target=self.prepare)
        self.prepare_thread.daemon = True
        self.prepare_thread.start()

    def prepare(self):
        tensor_files = os.listdir('data/training/tensors')
        train_data, target_data = [], []

        if len(tensor_files) == 0:
            print("No tensor files avaible. Please make them!")
            exit()

        while self.stopping == False:
            if len(self.batches) >= self.max_batches_loaded:
                time.sleep(0.2)
                continue
            
            f = random.choice(tensor_files)
            tensors = torch.load('data/training/tensors/' + f)

            for _ in range(0, random.randint(0, config.TRAIN_BATCH_SIZE - len(train_data))):
                choosen = random.choice(tensors)
                train_data.append(choosen[0])
                target_data.append(choosen[1])

            if len(train_data) >= config.TRAIN_BATCH_SIZE:
                self.batches.append((torch.stack(train_data), target_data))
                target_data = []
                train_data = []


    def stop(self):
        self.stopping = True
        self.batches = []

    def get_batch(self):
        while len(self.batches) == 0:
            time.sleep(0.5)

        b = self.batches.copy()[0]
        self.batches.pop(0)
        return [b]

def train_tomatoes(model : TomatoNetwork, epochs : int, batches_loaded_count = 20):
    if epochs <= 0:
        epochs = 10000000 #10 Mio.   

    #set the network in training mode
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.1) 
    loader = batch_loader(batch_size=config.TRAIN_BATCH_SIZE, max_batches_loaded=batches_loaded_count)
    loss = None
    best_Loss = None
    start_time = None

    try:           
        for epoch in range(epochs):
            for batch_number in range(get_total_batch_count()):
                batch_data = loader.get_batch()
                start_time = time.time()

                for batch in batch_data:                      
                    #iterate through every batch                                     
                    _input = Variable(batch[0])
                    _output = model(_input)
                    _target = Variable(torch.Tensor(batch[1]))                   

                    #optimize loop
                    criterion = nn.BCELoss()
                    optimizer.zero_grad()
                    model.zero_grad()

                    loss = criterion(_output, _target)
                    if best_Loss == None:
                        best_Loss = loss

                    loss.backward()                                     
                    optimizer.step()

                    #save only better results
                    if loss <= best_Loss:
                        torch.save(model, config.TOMATO_NETWORK_FILE_NAME)

                print(f'Batch_number: {batch_number}/{get_total_batch_count()}   Loss: {loss}   Epoch: {epoch}/{epochs}   Time: {time.time() - start_time}s')
                #print statements

    except KeyboardInterrupt:
        print("Exit(0): Keyboard Interrupt! Saving")
        torch.save(model, config.TOMATO_NETWORK_FILE_NAME)
    
    loader.stop()
    return model

def test_tomatoes(model : TomatoNetwork, tests : int):
    labels = os.listdir('data/validation')
    model.eval()

    succes = 0
    failed = 0

    for t in range(tests):
        for l in labels:
            images = os.listdir('data/validation/' + l)
            while len(images) > 0:
                f = random.choice(images)
                images.remove(f)

                img = cv2.imread('data/validation/' + l + '/' + f)
                _input = img_processing(img)
                _input.unsqueeze_(0)
                _output = tensor_to_highest(model(_input).squeeze(0))

                if _output == get_target(l):
                    succes += 1
                else:
                    failed += 1
                print(f'Test row: {t}  Current Succes: {succes}   Current Failed: {failed}   Total Images: {succes+failed}  Current Percentage: {succes / (succes + failed) * 100}%')

    return model


