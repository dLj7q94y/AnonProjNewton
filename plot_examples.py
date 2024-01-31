import argparse
import os
import utils.utils as utils
from utils.utils_plot import plot_example, save_plot
import matplotlib as mpl

mpl.rcParams.update({'font.size': 12, 'font.weight' : "bold"})


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--examples", help="The example to run.", nargs="+")
parser.add_argument("--all", help="Plots all of the included examples", action="store_true")
args = parser.parse_args()

ex = args.examples

if ex is None:
    ex=[]

if args.all:
    ex = ["L1_log_digits", "L1_log_MNIST", "L1_multi_CIFAR", "L1_MLP_fashionMNIST", "NNMF_image_noncvx", "NNMF_text_cosine"]

print("Plotting examples {}.".format(ex))

for e in ex:
    if "NNMF" in e:
        ops_weighting = (1, 1, 2)
    if "L1" in e:
        ops_weighting = (1, 1, 4)
    
    plot = plot_example(e, ops_weighting=ops_weighting)
    save_plot(plot, e)