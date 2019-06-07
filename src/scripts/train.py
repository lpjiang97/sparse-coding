import os
import sys
sys.path.insert(0, os.path.abspath('../../.'))
from tqdm import tqdm
import random
import torch
import torch.nn.functional as F
from src.model.PCNet import PCNet
from src.model.ImageDataset import load_all_patches
from src.utils.cmd_line import parse_args

### HYPERPARAMS ###
arg = parse_args()

# if use cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create net
pcnet = PCNet(arg.n_neuron, arg.size, R_epochs=arg.r_epoch, R_lr=arg.r_learning_rate, lmda=arg.reg)
# get data
X = load_all_patches(arg.batch_size, arg.size).to(device)
# train
for x in tqdm(range(arg.epoch), desc='training', total=arg.epoch):
    # sample patches
    idx = random.sample(range(X.shape[0]), arg.batch_size)
    image_batch = X[idx, :]
    # update
    pred = pcnet(image_batch)
    loss = ((image_batch - pred) ** 2).sum()
    loss.backward()
    # update U
    pcnet.U.data.sub_(arg.learning_rate * pcnet.U.grad.data)
    # zero grad
    pcnet.zero_grad()
    # normalize
    with torch.no_grad():
        pcnet.U = F.normalize(pcnet.U, dim=1)
    pcnet.U.requires_grad_(True)
    # save
    if x % 100 == 0:
        torch.save(pcnet, f"../../trained_models/model_epoch-{x}_N-{arg.batch_size}_K-{arg.n_neuron}_M-{arg.size}_lmda-{arg.reg}_Rlr_{arg.r_learning_rate}_Ulr_{arg.learning_rate}.pth")

torch.save(pcnet, f"../../trained_models/model_epoch-{x+1}_N-{arg.batch_size}_K-{arg.n_neuron}_M-{arg.size}_lmda-{arg.reg}_Rlr_{arg.r_learning_rate}_Ulr_{arg.learning_rate}.pth")

