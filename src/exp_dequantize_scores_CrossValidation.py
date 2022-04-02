"""
Test script for ranking with multiple sources of information
- yushal@andrew.cmu.edu

We try to dequantize the scores. The goal is to recover a set of scores that are intuitive for the area chairs to interpretate, and incorporates the rankings as well. See the definition of Dataset_sub() for data generation

Quantization Validation version chooses lambda (in the objective of soft estimator: sum (y_ij - mean(y_ij))^2 + lambda * sum (y_ij - z_ij)^2 ). The validation set is created by further quantizing the dataset. 
"""

import numpy as np
import sys
sys.path.append('..')
import ipdb
import os 
import argparse
import random
import pickle
from time import time
from src.consensus_maximizer import *
from src.borda_subjective import * 
from utils.utils_funcs import *
from data_loader.dataset_subjective import *
import copy

'''Set up'''
parser = argparse.ArgumentParser()
parser.add_argument('--n_item', type=int, default=1, help='number of all item')
parser.add_argument('--num_tests', type=int, default=1, help='number of tests')
parser.add_argument('--num_scores', type=int, default=4)
parser.add_argument('--num_assigns', type = int, default=4)
parser.add_argument('--sigma', type = float, default=1.0, help='noise std')
parser.add_argument('--nu', type=float, default=0.05, help='comparison_margin')
parser.add_argument('--dist', type=str, default="uniform", help='dist of true data scores')
parser.add_argument('--save', dest = "do_save",action='store_true', default=False)
parser.add_argument('--variance_loss', dest = "toggle_variance_loss",action='store_true', default=False)

args = parser.parse_args()
print(args)
configs = {}
file_dir = "results/"
stamp = 'SIM-{:d}-{:d}-{:.2f}'.format(args.num_scores, args.num_assigns, args.sigma)
test_folder = os.path.join(file_dir,'%s'%(stamp))
if args.do_save:
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

""" Test """
check_time = time()
results = {}
results["soft"] = []
results["rating"] = []
results["borda same"] = []

errors = {}
for key in results.keys():
    errors[key] = []

dataset = Dataset_sub(args.n_item, configs)
qv_lambdas = [] # quantiztion validation: selected lambda
for trial in range(args.num_tests):
    print("trial {}".format(trial))
    np.random.seed(trial)
    random.seed(trial)
    dataset.get_combined_data(num_scores=args.num_scores, num_assigns=args.num_assigns, savepath=None, data_dist = args.dist, noise_std = args.sigma) 
    results["rating"].append(dataset.paper_to_rating)
    # everything else is default
    if args.do_save:
        file_name = os.path.join(test_folder,'dataset_dict_'+str(trial))
        file = open(file_name, 'wb')
        pickle.dump(dataset.__dict__, file)
        file.close()
    
    cv_dataset = copy.deepcopy(dataset)
    # for attr in dataset.__dict__.keys():
    #     cv_dataset.attr = dataset.__dict__[attr]
    """Further quantization"""
    cv_dataset.paper_to_score = cv_dataset.paper_to_rating
    cv_dataset.paper_to_rating = np.ceil(cv_dataset.paper_to_score/2)

    cv_dataset.reviewer_to_score = cv_dataset.reviewer_to_rating
    cv_dataset.reviewer_to_rating = np.ceil(cv_dataset.reviewer_to_rating/2)
    
    """Unlike original dataset, the further quantized one needs partial constraint """
    cv_dataset.reviewer_to_list_of_groups = []
    for reviewer_id in range(cv_dataset.reviewer_to_rating.shape[0]):
        z = cv_dataset.reviewer_to_score[reviewer_id]
        list_of_groups = [np.where(z==v)[0] for v in sorted(np.unique(z),reverse=True)]
        cv_dataset.reviewer_to_list_of_groups.append(list_of_groups)

    consensus_maximizer = ConsensusMaximizer()
    """ soft constraint choosing lambda with cross validation """
    print("Soft estimator:")
    est_ys_soft = np.zeros_like(dataset.paper_to_score)
    range_lambda = [np.exp(i/4) for i in range(40)]
    cv_err = []
    cv_l2_dist = []
    for lambda_value in range_lambda:
        cv_sol_soft, max_obj = consensus_maximizer.solve_soft_auto(dataset=cv_dataset, comparison_margin=args.nu, score_lambda = lambda_value, toggle_variance_loss = args.toggle_variance_loss, comp_constraint = "partial")
        cv_sol_soft = np.round(cv_sol_soft,decimals=4)
        err = cv_dataset.partial_kendalltau_dist(cv_sol_soft.flatten(), regularize=True)
        
        cv_err.append(err)
        print("lambda = {}, kendall-tau err = {:.3f}".format(lambda_value, err))


    best_idx = np.argmin(cv_err)
    best_lambda = range_lambda[best_idx]
    qv_lambdas.append(best_lambda)

    print("best lambda value is {}".format(best_lambda))

    est_ys_soft, max_obj = consensus_maximizer.solve_soft_auto(dataset=dataset, comparison_margin=args.nu, score_lambda = best_lambda, toggle_variance_loss = args.toggle_variance_loss, comp_constraint = "random")
    print("when lam={}, max |y-z|={}".format(best_lambda, max_obj))
    results["soft"].append(est_ys_soft)

    """Borda count estimations of y"""
    est_ys_borda_same = borda_sub(dataset, args.nu, "same", toggle_partial_ranking=False)
    results["borda same"].append(est_ys_borda_same)


    true_rank = get_rank(dataset.paper_to_score.flatten())
    for idx, key in enumerate(results.keys()):
        sol = results[key][trial]
        sol = np.round(sol,decimals=4)
        if key.lower()=="mle": continue
        assert sol.shape == dataset.paper_to_score.shape
        """ Treat ties as 0.5 """
        err_kt = kendalltau_dist(dataset.paper_to_score.flatten(), sol.flatten(),True, False)
        errors[key].append(err_kt)

    print("used time: {}".format(time() - check_time))
    check_time = time()


print(errors)
print("selected lambdas are {}".format(qv_lambdas))

if args.do_save:
    file_name = os.path.join(test_folder,'results_dict')
    file = open(file_name, 'wb')
    pickle.dump(results, file)
    file.close()

    file_name = os.path.join(test_folder,'errors_dict')
    file = open(file_name, 'wb')
    pickle.dump(errors, file)
    file.close()

    print("saved errors to %s" % file_name)

    file_name = os.path.join(test_folder,'args_dict')
    file = open(file_name, 'wb')
    pickle.dump(args.__dict__, file)
    file.close()

    file_name = os.path.join(test_folder,'lambdas_dict')
    file = open(file_name, 'wb')
    pickle.dump(qv_lambdas, file)
    file.close()

    file_name = os.path.join(test_folder, 'cv_errors_dict')
    file = open(file_name, 'wb')
    pickle.dump(cv_err, file)
    file.close()