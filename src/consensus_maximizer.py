"""
Subjective estimator that aims to dequantize scores with (partial) rankings and quantized scores

The current implementation includes 2 forms of consensus maximizer, the difference is in the tie-breaking constraint. Hard V.S soft. 
"""
import numpy as np
import sys
sys.path.append('..')
import math
import ipdb
from scipy.stats import norm
from utils.utils_funcs import *

import cvxpy as cp

# import torch
# import torch.nn.functional as F
# from torch import optim
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
# Tensor = FloatTensor
# print("USE", torch.cuda.device_count(), "GPUs!")
# torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

class ConsensusMaximizer():
    """
    The estimator that maximizes the consensus among different reviews for each paper
    """
    def __init__(self):
        self.est_y = None

    def solve_soft_auto(self, dataset, comparison_margin=0.1, score_lambda = 0.01, toggle_variance_loss=False, comp_constraint = "random"):
        """
        Using CVXPY to specify obj and constraint(need to derive from reviewer feedbacks)
        """
        num_papers = dataset.paper_to_rating.shape[0]
        num_reviews = dataset.paper_to_rating.shape[1]
        est_y = cp.Variable((num_papers, num_reviews))
        if toggle_variance_loss:
            mean_ys = cp.sum(est_y, axis=1, keepdims=True)/num_reviews
            var_loss = cp.sum_squares(est_y - mean_ys@np.ones((1,num_reviews)))
            bin_loss = cp.sum_squares(est_y - dataset.paper_to_rating)
            obj = cp.Minimize(var_loss + score_lambda*bin_loss)
        else:
            raise ValueError("Not implemented yet")
        constraints = []

        ratings_matrix = np.asarray(dataset.paper_to_rating)
        constraints += [est_y<=ratings_matrix+0.5]
        constraints += [est_y>=ratings_matrix-0.5]


        if comp_constraint=="random":
            for reviewer_id, paper_ids in enumerate(dataset.reviewer_to_paper):
                positions = np.where(dataset.paper_to_reviewer[paper_ids]==reviewer_id)[1] # elem z: the position of this reviewer in the zth paper (in paper_ids)
                constraints += [est_y[paper_ids[j_idx], positions[j_idx]] >= (est_y[paper_ids[j_idx+1], positions[j_idx+1]]+comparison_margin) for j_idx in range(len(paper_ids)-1)]
        
        elif comp_constraint=="partial":
            for reviewer_id, paper_ids in enumerate(dataset.reviewer_to_paper):
                positions = np.where(dataset.paper_to_reviewer[paper_ids]==reviewer_id)[1]
                # paper_id_to_position = {paper_ids[i]: positions[i] for i in range(len(paper_ids))} # paper_id : position of this reviewer
                list_of_groups = dataset.reviewer_to_list_of_groups[reviewer_id]
                for group_idx in range(len(list_of_groups)-1):
                    pairs = np.array(np.meshgrid(list_of_groups[group_idx],list_of_groups[group_idx+1])).T.reshape([-1,2]) # each row is a pair of papers, one in each of the two groups
                    for j, j_prime in pairs:
                        constraints += [est_y[paper_ids[j], positions[j]] - est_y[paper_ids[j_prime], positions[j_prime]]>= comparison_margin]
        else:
            raise ValueError("Not implemented yet, check comp_constraint: {}".format(comp_constraint))
        
        soft_prob = cp.Problem(obj, constraints)
        print("Is DPP? ", soft_prob.is_dcp(dpp=True))
        print("Is DCP? ", soft_prob.is_dcp(dpp=False))
        soft_prob.solve(solver=cp.CVXOPT, verbose=False, feastol=1e-6)  # Returns the optimal value.
        print("status:", soft_prob.status)
        print("optimal value", soft_prob.value)
        np_est_y = est_y.value

        """ DEPRECATED: Check whether constraints are violated """
        max_diff = np.max(np.abs(np_est_y - dataset.paper_to_rating))
        return np_est_y, max_diff
