import argparse


parser = argparse.ArgumentParser(description="Template")
# model
parser.add_argument('-N', '--batch_size', default=2000, type=int, help="Batch size")
parser.add_argument('-K', '--n_neuron', default=100, type=int, help="The number of neurons")
parser.add_argument('-M', '--size', default=10, type=int, help="The size of receptive field")
# training
parser.add_argument('-e', '--epoch', default=5000, type=int, help="Number of Epochs")
parser.add_argument('-re', '--r_epoch', default=300, type=int, help="Number of Epochs for ISTA")
parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help="Learning rate")
parser.add_argument('-rlr', '--r_learning_rate', default=1e-2, type=float, help="Learning rate for ISTA")
parser.add_argument('-lmda', '--reg', default=5e-3, type=float, help="LSTM hidden size")


# Parse arguments
def parse_args():
	return parser.parse_args()