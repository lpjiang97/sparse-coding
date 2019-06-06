import os
import sys
sys.path.insert(0, os.path.abspath('../../.'))
import torch
import matplotlib.pyplot as plt
from src.utils.cmd_line import parse_args


### choose the model to load ###
arg = parse_args()

# load model
model = torch.load(f"../../trained_models/model_epoch-{arg.epoch}_N-{arg.batch_size}_K-{arg.n_neuron}_M-{arg.size}_lmda-{arg.reg}_Rlr_{arg.r_learning_rate}_Ulr_{arg.learning_rate}.pth")
fig, axes = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True)
for i in range(10):
    for j in range(10):
        ax = axes[i, j]
        ax.imshow(model.U[i * 10 + j, :].data.numpy().reshape(-1,10), cmap='gray')
fig.set_size_inches(15, 15)
fig.savefig(f"../../trained_models/model_epoch-{arg.epoch}_N-{arg.batch_size}_K-{arg.n_neuron}_M-{arg.size}_lmda-{arg.reg}_Rlr_{arg.r_learning_rate}_Ulr_{arg.learning_rate}.png", dpi=300)
