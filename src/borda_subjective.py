"""
Subjective estimator that aims to recover the scores that reviewers would have given to papers if not for the quantization.

The current implementation includes forms of borda count baselines, where for a fixed amount is added to the quantized scores based on their rank position. 
"""

import numpy as np
import sys
sys.path.append('..')
import math
import ipdb
from scipy.stats import norm
from utils.utils_funcs import *
from sklearn.isotonic import IsotonicRegression

def borda_sub(dataset, comparison_margin, mode, toggle_partial_ranking=False):
    num_reviews = dataset.reviewer_to_rating.shape[1]
    if toggle_partial_ranking==False:
        if mode == "all": # all scores are adjusted 
            adjust_amount = np.linspace((num_reviews-1)*comparison_margin, 0, num=num_reviews)
        elif mode == "same": # only same scored are adjusted
            adjust_amount = np.zeros(num_reviews)
        else:
            raise ValueError("Check mode={}".format(str(mode)))
        est_y = dataset.paper_to_rating.copy()
        for reviewer_id, paper_ids in enumerate(dataset.reviewer_to_paper):
            positions = np.where(dataset.paper_to_reviewer[paper_ids]==reviewer_id)[1]
            if mode == "all":
                est_y[paper_ids, positions] = dataset.reviewer_to_rating[reviewer_id] + adjust_amount
            elif mode == "same":
                list_of_group = [np.where(dataset.reviewer_to_rating[reviewer_id]==v)[0] for v in sorted(np.unique(dataset.reviewer_to_rating[reviewer_id]),reverse=True)]
                for group in list_of_group:
                    if len(group)<=1:
                        continue
                    adjust_amount = np.linspace((len(group)-1)*comparison_margin, 0, num=len(group))
                    score = dataset.reviewer_to_rating[reviewer_id][group] + adjust_amount
                    score -= np.mean(score) - dataset.reviewer_to_rating[reviewer_id][group[0]]

                    est_y[paper_ids[group], positions[group]] = score
    else: 
        print("Partial comparison constraint ON")
        est_y = dataset.paper_to_rating.copy()
        for reviewer_id, paper_ids in enumerate(dataset.reviewer_to_paper):
            positions = np.where(dataset.paper_to_reviewer[paper_ids]==reviewer_id)[1]
            if mode == "all":
                total_groups = len(dataset.reviewer_to_list_of_groups[reviewer_id])
                for idx, group in enumerate(dataset.reviewer_to_list_of_groups[reviewer_id]):
                    est_y[paper_ids[group], positions[group]] = dataset.reviewer_to_rating[reviewer_id][group] + comparison_margin*(total_groups-1-idx)
            elif mode == "same": # same bin
                # create vector for group number
                group_number = np.zeros_like(dataset.reviewer_to_rating[reviewer_id])
                for idx, group in enumerate(dataset.reviewer_to_list_of_groups[reviewer_id]):
                    group_number[group] = idx
                list_of_tie_group = [np.where(dataset.reviewer_to_rating[reviewer_id]==v)[0] for v in sorted(np.unique(dataset.reviewer_to_rating[reviewer_id]),reverse=True)]
                for tied in list_of_tie_group:
                    if len(np.unique(group_number[tied]))==1:
                        continue # only one group in this set of tied papers
                    coefficients = np.max(group_number[tied]) - group_number[tied]
                    score = dataset.reviewer_to_rating[reviewer_id][tied] + comparison_margin*coefficients
                    score -= np.mean(score) - dataset.reviewer_to_rating[reviewer_id][tied[0]]

                    est_y[paper_ids[tied], positions[tied]] = score
    return est_y

def borda_even(dataset):
    """
    DEPRECATED A naive baseline where tied scores are pushed to evenly-spaced values.
    """
    est_y = dataset.paper_to_rating.copy()
    for reviewer_id, paper_ids in enumerate(dataset.reviewer_to_paper):
        positions = np.where(dataset.paper_to_reviewer[paper_ids]==reviewer_id)[1]
        list_of_group = [np.where(dataset.reviewer_to_rating[reviewer_id]==v)[0] for v in sorted(np.unique(dataset.reviewer_to_rating[reviewer_id]),reverse=True)]
        for group in list_of_group:
            if len(group)<=1:
                continue
            adjusted_scores = np.linspace(start= dataset.reviewer_to_rating[reviewer_id][group[1]]-0.5, stop= dataset.reviewer_to_rating[reviewer_id][group[1]]+0.5, num = len(group)+2, endpoint=True)[::-1]
            est_y[paper_ids[group], positions[group]] = adjusted_scores[1:-1]
    return est_y
