import torch
import torch.nn as nn
import torch.nn.functional as F
import load as L
import numpy as np
import copy
import random
import time
import argparse
import torch.utils.tensorboard
import os
from tqdm import tqdm

from mlp import MLP
# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--seed",type=int,default=-1,help="Set random seed to ensure reproducible results,-1 for no seed")
parser.add_argument("--cuda",type=str,default="default",choices=["default","cpu",*map(str,range(0,100))],help="which device to use")
parser.add_argument("--valid-rate",type=float,default=0.9)
parser.add_argument("--batch-size",type=int,default=64)
parser.add_argument("--load-module",type=bool,default=False)
parser.add_argument("--lr",type=float,default=0.1)
parser.add_argument("--gamma",type=float,default=0.9995)
parser.add_argument("--log-per-iteration",type=int,default=100)
parser.add_argument("--epoch",type=int,default=500)
parser.add_argument("--num-epoch-to-validation",type=int,default=10)
parser.add_argument("--num-epoch-to-save",type=int,default=5)
args = parser.parse_args()



# Set random seed to ensure reporducible results
if args.seed >= 0: 
    SEED = args.seed 
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# CUDA device
if args.cuda == "default":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
elif args.cuda == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device("cuda:"+str(args.cuda))

# Dataset
train_set, test_set = L.load_dataset() 
VALID_RATE = args.valid_rate 
train_set, valid_set = torch.utils.data.random_split(train_set, [int(len(train_set)*VALID_RATE), len(train_set)-int(len(train_set)*VALID_RATE)])
train_set_len, valid_set_len, test_set_len = len(train_set), len(test_set), len(valid_set)

# DataLoader
BATCH_SIZE = args.batch_size 
train_iterator = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE,drop_last=True) # Warning: Setting drop_last=False may cause log writter write wrong things. Take care
valid_iterator = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE,drop_last=True)
test_iterator  = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,drop_last=True)

# Module
LOAD_MODULE = args.load_module 
LOAD_MODULE_PATH = "./checkpoint/model.pth"
if not LOAD_MODULE:
    model = MLP() # TODO: check parameters given to NN such as input_dim,output,dim
else:
    model = torch.load(LOAD_MODULE_PATH)
    

# Loss function 
loss_func = nn.CrossEntropyLoss()

# Optim
LEARNING_RATE = args.lr
optim = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Scheduler
GAMMA = args.gamma
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim,gamma=GAMMA, last_epoch=-1)

# Log
writer = torch.utils.tensorboard.SummaryWriter()
LOG_PER_ITERATION = args.log_per_iteration  # it may log more when not divisible, but never mind.
log_iter_num = (train_set_len // BATCH_SIZE) // LOG_PER_ITERATION
if log_iter_num <= 0:
    log_iter_num = 1


# train
EPOCH = args.epoch 
NUM_EPOCH_TO_VALIDATION = args.num_epoch_to_validation  # set this <= 0 to avoid validation during training.
NUM_EPOCH_TO_SAVE = args.num_epoch_to_save 
MODEL_SAVE_PATH = "./checkpoint/"
if not os.path.exists(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)
model = model.to(device).float()  # or .double()
try:
    sum_of_loss = 0
    sum_loss_counter = 0
    for epoch in tqdm(range(EPOCH)):
        # Gradient Descent Process
        for i,(xi,yi) in enumerate(train_iterator):                 # TODO: How is the data look like ? 
            xi, yi = xi.to(device).float(), yi.to(device).long()   # TODO: How is the data look like ? .float() may be changed to .double() / .long() 
            yi_ = model(xi)                                         # TODO: How to manipulate data using model ? 
            loss = loss_func(yi_,yi)                                # TODO: How to calculate loss? 
            loss.backward()
            optim.step()
            optim.zero_grad()
            sum_of_loss += loss.item()
            sum_loss_counter += 1 
            if (i + 1) % log_iter_num == 0:
                writer.add_scalar("Loss/train",(sum_of_loss/(sum_loss_counter*BATCH_SIZE)),(epoch*BATCH_SIZE*(train_set_len//BATCH_SIZE)+i*BATCH_SIZE)) #(loss_of_these_counts_per_data, all_seen_data_number)
                sum_of_loss = 0
                sum_loss_counter = 0
        scheduler.step()
        # Validation During Train
        if NUM_EPOCH_TO_VALIDATION > 0 and (epoch + 1) % NUM_EPOCH_TO_VALIDATION == 0:
            with torch.no_grad():
                sum_of_loss = 0
                sum_loss_counter = 0
                # for classification only
                sum_correct_item = 0
                # end for classification only
                for i,(xi,yi) in enumerate(valid_iterator):
                    xi, yi = xi.to(device).float(), yi.to(device).long() 
                    yi_ = model(xi)
                    loss = loss_func(yi_,yi)
                    sum_loss_counter += 1 
                    sum_of_loss += loss.item() # For Regression 
                    sum_correct_item += torch.sum(torch.eq(yi,yi_.argmax(dim=1))).item() # For Classification
                writer.add_scalar("Loss/validation",(sum_of_loss/(sum_loss_counter*BATCH_SIZE)),epoch)   #TODO :How to Calculate validation item
                writer.add_scalar("Accuracy/validation",(sum_correct_item/(sum_loss_counter*BATCH_SIZE)),epoch) #TODO :How to Calculate validation item
        if NUM_EPOCH_TO_SAVE > 0 and (epoch + 1) % NUM_EPOCH_TO_SAVE == 0:
            torch.save(model,MODEL_SAVE_PATH+"_epoch_"+str(epoch)+"_"+str(model.__class__.__name__))
    torch.save(model,MODEL_SAVE_PATH+str(model.__class__.__name__)+str(time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime())))
except KeyboardInterrupt:     # TODO : Set to Exception to catch all exceptions
    torch.save(model,MODEL_SAVE_PATH+str(model.__class__.__name__))
    torch.save(model,MODEL_SAVE_PATH+"model.pth")
    

