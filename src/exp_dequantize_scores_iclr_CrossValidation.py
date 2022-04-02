"""
Test script for ranking with multiple sources of information: Simulations on ICLR data from PeerRead public repo. 
- yushal@andrew.cmu.edu

Current version: 
We aim to dequantize the review scores. The goal is to recover a set of scores that are intuitive for the area chairs to interpretate, and incorporates the rankings as well. 

See the definition of Dataset_sub_iclr() for data generation
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
from data_loader.dataset_peerread import *
import copy

'''Set up'''
parser = argparse.ArgumentParser()
parser.add_argument('--num_tests', type=int, default=1, help='number of tests')
parser.add_argument('--num_assigns', type = int, default=4)
parser.add_argument('--nu', type=float, default=0.05, help='comparison_margin')
parser.add_argument('--save', dest = "do_save",action='store_true', default=False)
parser.add_argument('--variance_loss', dest = "toggle_variance_loss",action='store_true', default=False)
parser.add_argument('--comp_constraint', type=str, default = None, help ='constraint style: "partial", "random" or "strict"')

args = parser.parse_args()
print(args)
configs = {}
file_dir = "results/"
if not os.path.isdir(file_dir):
    raise ValueError("Check file directory: {}".format(file_dir))
stamp = 'ICLR-{:d}'.format(args.num_assigns)
test_folder = os.path.join(file_dir,'%s'%(stamp))
if args.do_save:
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

""" Test  """
check_time = time()
results = {}
results["soft"] = []
results["borda same"] = []
results['rating'] = []

errors = {}
qv_lambdas = []
for key in results.keys():
    errors[key] = []

for trial in range(args.num_tests):
    print("trial {}".format(trial))
    np.random.seed(trial)
    random.seed(trial)

    dataset = Dataset_sub_iclr(reviewer_mode="random")
    # dataset.get_small_data(path="peerread_data/iclr_2017",comparison_constraint=args.comp_constraint)
    dataset.get_data(num_assigns=args.num_assigns, path="peerread_data/iclr_2017", comparison_constraint=args.comp_constraint)
    results["rating"].append(dataset.paper_to_rating)
    print("Baseline rating error is {}".format(dataset.partial_kendalltau_dist(dataset.paper_to_rating.flatten(), regularize=True)))
    
    if args.do_save:
        file_name = os.path.join(test_folder,'dataset_dict_'+str(trial))
        file = open(file_name, 'wb')
        pickle.dump(dataset.__dict__, file)
        file.close()

    cv_dataset = copy.deepcopy(dataset)
    # for attr in dataset.__dict__.keys():
    #     cv_dataset.attr = dataset.__dict__[attr]
    """Further quantization"""
    cv_dataset.original_scores = cv_dataset.paper_to_rating
    cv_dataset.reviewer_to_score = cv_dataset.reviewer_to_rating
    cv_dataset.paper_to_rating = np.ceil(cv_dataset.original_scores/2)
    cv_dataset.reviewer_to_rating = np.ceil(cv_dataset.reviewer_to_rating/2)
    if args.comp_constraint == "partial":
        cv_dataset.reviewer_to_list_of_groups = []
        for reviewer_id in range(cv_dataset.reviewer_to_rating.shape[0]):
            z = cv_dataset.reviewer_to_score[reviewer_id]
            list_of_groups = [np.where(z==v)[0] for v in sorted(np.unique(z),reverse=True)]
            cv_dataset.reviewer_to_list_of_groups.append(list_of_groups)
    else:
        raise ValueError(args.comp_constraint)


    consensus_maximizer = ConsensusMaximizer()

    """ soft constraint choosing lambda with cross validation """
    print("Soft estimator:")
    est_ys_soft = np.zeros_like(dataset.paper_to_rating)
    range_lambda = [np.exp(i/4) for i in range(40)]
    cv_err = []
    for lambda_value in range_lambda:
        cv_sol_soft, max_obj = consensus_maximizer.solve_soft_auto(dataset=cv_dataset, comparison_margin=args.nu, score_lambda = lambda_value, toggle_variance_loss = args.toggle_variance_loss, comp_constraint = args.comp_constraint)

        cv_sol_soft = np.round(cv_sol_soft,decimals=4)

        if args.comp_constraint=='random':
            est_rank_soft = get_rank(cv_sol_soft.flatten())
            true_rank = get_rank(cv_dataset.paper_to_score.flatten())
            err = kendalltau_dist(true_rank, est_rank_soft)
        elif args.comp_constraint=='partial':
            err = cv_dataset.partial_kendalltau_dist(cv_sol_soft.flatten(), regularize=True)
        else:
            raise ValueError("Check args.comp_constarint: {}".format(args.comp_constarint))
        
        cv_err.append(err)
        print("lambda = {}, kendall-tau err = {:.6f}".format(lambda_value, err))
    best_idx = np.argmin(cv_err)
    best_lambda = range_lambda[best_idx]
    qv_lambdas.append(best_lambda)
    print("best lambda value is {}".format(best_lambda))

    est_ys_soft, max_obj = consensus_maximizer.solve_soft_auto(dataset=dataset, comparison_margin=args.nu, score_lambda = best_lambda, toggle_variance_loss = args.toggle_variance_loss, comp_constraint = args.comp_constraint)
    results["soft"].append(est_ys_soft)
    
    """Borda count estimations of y"""
    est_ys_borda_same = borda_sub(dataset, args.nu, "same",toggle_partial_ranking=(args.comp_constraint=="partial"))
    results["borda same"].append(est_ys_borda_same)


    print("used time: {}".format(time() - check_time))
    check_time = time()
    if args.comp_constraint=='random':
        true_rank = dataset.y_ranking.flatten()
        for idx, key in enumerate(results.keys()):
            sol = results[key][trial]
            sol = np.round(sol,decimals=4)
            if key.lower()=="mle": continue
            assert sol.shape == dataset.original_scores.shape
            err_kt = []
            for _ in range(50): # take into account the randomness
                rank = get_rank(sol.flatten())
                err_kt.append(kendalltau_dist(true_rank, rank))
            errors[key].append(np.mean(err_kt))
    elif args.comp_constraint=='partial':
        for idx, key in enumerate(results.keys()):
            sol = results[key][trial]
            sol = np.round(sol,decimals=4)
            if key.lower()=="mle": continue
            assert sol.shape == dataset.original_scores.shape
            err = dataset.partial_kendalltau_dist(sol.flatten(), regularize=True)
            errors[key].append(err)
        
print("errors are {}".format(errors))
print("selected lambdas are: {}".format(qv_lambdas))
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