# %%
#### L1 Multinomial Results
from utils.utils_plot import plot_full_results, iteration_oracle_count, result_accuracy, iteration_oracle_count
import utils.utils as utils
import os 

# Input the save folder
#file = "L1_log_MNIST"
file = "L1_multi_CIFAR"

ops_weighting = (1, 1, 4)

plot_full_results(file, ops_weighting=ops_weighting, eg_hline=1e-8)
iteration_oracle_count(file, ops_weighting=ops_weighting)
result_accuracy(file)

# %%
#### MLP Results
from utils.utils_plot import plot_full_results, iteration_oracle_count, result_accuracy
import utils.utils as utils

file = "L1_MLP_fashionMNIST"

ops_weighting = (1, 1, 4)

plot_full_results(file, ops_weighting=ops_weighting, eg_hline=1e-8)
iteration_oracle_count(file, ops_weighting=ops_weighting)
result_accuracy(file)


# %%
#### Image NNMF results
import numpy as np
from utils.utils_plot import plot_full_results, iteration_oracle_count, result_loss_sparity, plot_image_representation, save_plot
import utils.utils as utils

file = "NNMF_image_noncvx"

ops_weighting = (1, 1, 2)

plot_full_results(file, ops_weighting=ops_weighting, eg_hline=1e-8)
iteration_oracle_count(file, ops_weighting=ops_weighting)
result_loss_sparity(file)

# Generate a plot of the low rank representation generated by NNMF, H.
# Warning these are very scary!
# ensure r = n_col*n_row  
fig_mr = plot_image_representation(file, n_col=5, n_row=2)
save_plot(fig_mr, file+"_MR_H_faces")

fig_pgm = plot_image_representation(file, n_col=5, n_row=2, example="ProxGradMomentum.json")

# %% 
#### Text NNMF results. 

import numpy as np
from utils.utils_plot import plot_full_results, iteration_oracle_count, result_loss_sparity, plot_top_words, save_plot
import utils.utils as utils

file = "NNMF_text_cosine"
ops_weighting = (1, 1, 2)

plot_full_results(file, ops_weighting=ops_weighting, eg_hline=1e-8)
iteration_oracle_count(file, ops_weighting=ops_weighting)
result_loss_sparity(file)

# Generate a plot the top words in each "cluster" generated by NNMF. This corresponds to the largest values in each "document", i.e., row of H.
 
# make sure n_features is the same as the tfidf vectorizer used to generate the data.
# we see that we generate a nontrivial clustering of the topics.
fig_mr = plot_top_words(file, n_top_words=10, n_features=1000)
save_plot(fig_mr, file+"_MR_H_top_words")

# %%
from utils.utils_plot import plot_example_termination_conditions, save_plot
import utils.utils as utils

folder = "L1_multi_CIFAR"
ops_weighting = (1, 1, 4)
eg = 1e-8
fig = plot_example_termination_conditions(folder, ops_weighting=ops_weighting, hline=eg)
save_plot(fig, folder + "_local")

folder = "L1_log_MNIST"
ops_weighting = (1, 1, 4)
eg = 1e-8
fig = plot_example_termination_conditions(folder, ops_weighting=ops_weighting, hline=eg)
save_plot(fig, folder + "_local")

# %%
