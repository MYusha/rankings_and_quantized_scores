import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import seaborn as sns
import sys
sys.path.append('..')
import os
import numpy as np
import random
import math
from math import exp
import ipdb
import pickle
from utils.utils_funcs import *
from collections import OrderedDict
import pandas
import matplotlib.ticker as ticker
import itertools

np.random.seed(0)
random.seed(0)
sns.set_style("whitegrid")
regular_font_size = 18

folder = "results/"

def plot_lambda_histogram(file_dir, title):
    """ Plot histogram for selected values of lambda """
    file = open(folder+file_dir+"/lambdas_dict", "rb")
    qv_lambdas = pickle.load(file)
    file.close()
    print("largest data is {}".format(np.max(qv_lambdas)))
    fig, ax = plt.subplots()
    newbins = np.linspace(0,2000,50)
    sns.histplot(data=qv_lambdas, stat="probability", bins=newbins)
    plt.ylabel('Frequency', fontsize=regular_font_size)
    plt.xlabel('Values of lambdas', fontsize=regular_font_size)
    plt.ylim(0, 1)
    ax.tick_params(axis='both', labelsize=regular_font_size)
    # ax.set_title(title)
    fig.tight_layout()
    plt.show()

def calculate_ties_percentage(vector):
    vector = np.round(vector,decimals=4)
    diff_matrix = np.expand_dims(vector, axis=1) - np.expand_dims(vector, axis=0) # element {i, j} = vector(i) - vector(j)
    assert diff_matrix.shape[0] == diff_matrix.shape[1] # assert square
    ind = np.triu_indices(n=diff_matrix.shape[0], k=1) # only coordinates (i,j) s.t. i<j
    num_ties = len(np.where(diff_matrix[ind]==0)[0]) # number of such pairs
    percentage_ties = float(num_ties)/len(ind[0]) # total number of ties
    return percentage_ties*100

def print_difference(a, b, type = "percent"):
    assert type in ["percent", "value"]
    diff = a - b
    diff = float(diff)
    if type == "percent":
        if diff>=0: # a>b
            print("a is larger than b by {} percent".format((diff/b)*100))
        if diff<0: # a<b
            print("a is less than b by {} percent".format((-diff/b)*100))
    if type == "value":
        print("a - b = {}".format(diff))


################################
#### Put file names and labels here to visualize results ##
file_dir_list = ["ICLR-2", "ICLR-3", "ICLR-6"] # ICLR
file_labels = ["num_assigns", "2", "3", "6"]
################################

add_inset = False
do_l2 = False

labels = file_labels[1:]
keys = ["soft", "borda same", "rating"]
keys_to_color = {"soft": "blue", "borda same": "purple", "rating":"coral"}
keys_to_fmt = {"soft": "o-", "borda same": "v--", "rating":"P-."}
keys_to_labels = {"soft": "Proposed algorithm", "borda same": "BRE-adjusted-scores", "rating": "Quantized scores"}
label_to_names = {"num_scores": "Number of reviewers per paper", 
"num_assigns": "Number of papers per reviewer", 
"sigma": "The standard deviation parameter $\sigma$", 
"nu": "$\epsilon$" 
}


""" Plot error mean and std for several settings and methods """
error_list = []
fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
x = [float(value) for value in file_labels[1:]]
for idx_key, key in enumerate(keys):
    # key -> estimator
    error_mean = [] # error for this method, over different settings
    error_std = []

    l2_mean = []
    l2_std = []
    for idx_file, file_dir in enumerate(file_dir_list):
        """ Load file """
        file = open(folder+file_dir+"/results_dict", "rb")
        results = pickle.load(file)
        file.close()

        """ check settings are correct """
        args = pickle.load(open(folder+file_dir+"/args_dict", "rb"))
        if str(args[file_labels[0]]) != file_labels[idx_file+1]:
            raise ValueError("Check that args is {}".format(str(args[file_labels[0]])))
        
        """ load error """
        error = pickle.load(open(folder+file_dir+"/errors_dict", "rb"))
        num_trial = len(results["soft"])
        error_mean.append(np.mean(error[key]))
        error_std.append(np.std(error[key]))
        
        if do_l2:
            cur_setting_l2 = []
            dataset = pickle.load(open(folder+file_dir+"/dataset_dict_"+str(0), "rb"))
            """ ICLR: Load dataset for l2 error """
            if "original_scores" in dataset.keys():
                print("this is ICLR dataset")
                for trial in range(num_trial):
                    dataset = pickle.load(open(folder+file_dir+"/dataset_dict_"+str(trial), "rb"))
                    true_ys = dataset['original_scores']
                    sol = results[key][trial]
                    # if trial == 0:
                    #     print(np.max(true_ys), np.min(true_ys))
                    #     print(np.max(sol), np.min(sol))
                    sol = np.round(sol*2 -0.5)
                    # sol = sol + np.round(sol)-0.5
                    assert sol.shape == dataset["original_scores"].shape
                    cur_setting_l2.append(np.linalg.norm((sol-true_ys).flatten()))
            """ SIM: Load dataset for l2 error """
            if "paper_to_score" in dataset.keys():
                print("this is synthetic dataset")
                for trial in range(num_trial):
                    dataset = pickle.load(open(folder+file_dir+"/dataset_dict_"+str(trial), "rb"))
                    true_ys = dataset['paper_to_score']
                    sol = results[key][trial]
                    assert sol.shape == dataset["paper_to_score"].shape
                    cur_setting_l2.append(np.linalg.norm((sol-true_ys).flatten()))
            
            l2_mean.append(np.mean(cur_setting_l2))
            l2_std.append(np.std(cur_setting_l2))

        # """ Print percentage of ties in solution """
        num_trials = len(results[key])
        percentage_list = []
        for t in range(num_trials):
            percentage = calculate_ties_percentage(results[key][t].flatten())
            percentage_list.append(percentage)
        print("Average percentage of ties of estimator {} is {} when {} = {}".format(keys_to_labels[key], np.mean(percentage_list), file_labels[0], file_labels[idx_file+1]))
    if do_l2:
        plt.errorbar(x, l2_mean, yerr=l2_std/np.sqrt(num_trial), fmt=keys_to_fmt[key], color=keys_to_color[key], ecolor=keys_to_color[key], markersize=12, elinewidth=3, capsize=4, linewidth=3, label=keys_to_labels[key])
    else:
        plt.errorbar(x, error_mean, yerr=error_std/np.sqrt(num_trial), fmt=keys_to_fmt[key], color=keys_to_color[key], ecolor=keys_to_color[key], markersize=12, elinewidth=3, capsize=4, linewidth=3, label=keys_to_labels[key])

ax.set_xlabel(label_to_names[file_labels[0]], fontsize=regular_font_size)
if do_l2:
    ax.set_ylabel('$\ell_2$ Errors', fontsize=regular_font_size)
else:
    ax.set_ylabel('Normalized Kendall-tau Errors', fontsize=regular_font_size)
ax.set_ylim(0)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.tick_params(axis='both', labelsize=regular_font_size)
ax.legend(loc="best", fontsize=regular_font_size-2, handlelength=4)

if add_inset ==True:
    inset_ax = fig.add_axes([0.35, 0.4, 0.4, 0.4]) # X, Y, width, height
    # inset_ax = fig.add_axes([0.55, 0.4, 0.35, 0.35]) # X, Y, width, height
    inset_ax.set_title('zoom in view',fontsize=regular_font_size-2)
    for idx_key, key in enumerate(keys):
        # key -> estimator
        error_mean = [] # error for this method, over different settings
        error_std = []

        l2_mean = []
        l2_std = []
        for idx_file, file_dir in enumerate(file_dir_list):
            """ Load file """
            file = open(folder+file_dir+"/results_dict", "rb")
            results = pickle.load(file)
            file.close()

            """ check settings are correct """
            args = pickle.load(open(folder+file_dir+"/args_dict", "rb"))
            if str(args[file_labels[0]]) != file_labels[idx_file+1]:
                raise ValueError("Check that args is {}".format(str(args[file_labels[0]])))
            
            """ load error """
            error = pickle.load(open(folder+file_dir+"/errors_dict", "rb"))
            num_trial = len(results["soft"])
            error_mean.append(np.mean(error[key]))
            error_std.append(np.std(error[key]))

            
            cur_setting_l2 = []
            """ ICLR: Load dataset for l2 error """
            if "original_scores" in dataset.keys():
                print("this is ICLR dataset")
                for trial in range(num_trial):
                    dataset = pickle.load(open(folder+file_dir+"/dataset_dict_"+str(trial), "rb"))
                    true_ys = dataset['original_scores']
                    sol = results[key][trial]
                    sol = sol + np.round(sol)-0.5
                    assert sol.shape == dataset["original_scores"].shape
                    cur_setting_l2.append(np.linalg.norm((sol-true_ys).flatten()))
            """ SIM: Load dataset for l2 error """
            if "paper_to_score" in dataset.keys():
                print("this is synthetic dataset")
                for trial in range(num_trial):
                    dataset = pickle.load(open(folder+file_dir+"/dataset_dict_"+str(trial), "rb"))
                    true_ys = dataset['paper_to_score']
                    sol = results[key][trial]
                    assert sol.shape == dataset["paper_to_score"].shape
                    cur_setting_l2.append(np.linalg.norm((sol-true_ys).flatten()))
            
            l2_mean.append(np.mean(cur_setting_l2))
            l2_std.append(np.std(cur_setting_l2))
        inset_ax.errorbar(x, l2_mean, yerr=l2_std/np.sqrt(num_trial), fmt=keys_to_fmt[key], color=keys_to_color[key], ecolor=keys_to_color[key], markersize=12, elinewidth=3, capsize=4, linewidth=3, label=keys_to_labels[key])

    # set axis tick locations
    inset_ax.set_xticks(x)
    inset_ax.set_xticklabels(labels)
    inset_ax.tick_params(axis='both', labelsize=regular_font_size-2)
    # inset_ax.set_xlim(xmin=5.5, xmax=6.5)
    # inset_ax.set_ylim(5.2, 5.7)


fig.tight_layout()
plt.show()

""" Plot histogram for distribution of lambdas"""
# for idx_file, file_dir in enumerate(file_dir_list):
#     title = "Distribution of selected $\lambda$ by quantization-validation when {} = {}".format(file_labels[0], file_labels[idx_file+1])
#     plot_lambda_histogram(file_dir, title)

# # fig, ax = plt.subplots()
# # bins = np.linspace(0, 8500, 18)
# # for idx_file, file_dir in enumerate(file_dir_list):
# #     file = open(folder+file_dir+"/lambdas_dict", "rb")
# #     qv_lambdas = pickle.load(file)
# #     file.close()
# #     plt.hist(qv_lambdas, bins, alpha=0.5, label=str(idx_file))

# # plt.ylabel('Probability', fontsize=15)
# # plt.xlabel('Values of lambdas', fontsize=15)
# # ax.legend(loc='lower left', fontsize=15)
# # fig.tight_layout()
# # plt.show()


""" Calculate decrease in error """
# # print_difference(a, b, type = "percent")
for idx_file, file_dir in enumerate(file_dir_list):
    print("when {} = {}:".format(file_labels[0],file_labels[idx_file+1]))

    file = open(folder+file_dir+"/results_dict", "rb")
    results = pickle.load(file)
    file.close()

    error = pickle.load(open(folder+file_dir+"/errors_dict", "rb"))
    soft_error = np.mean(error["soft"])
    base_error = np.mean(error["borda same"])
    print(soft_error, base_error)
    print_difference(soft_error, base_error, "percent")
