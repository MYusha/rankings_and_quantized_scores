"""
Data loader Class, directly generates data
Called by test scripts only

This is the class that generates SUBJECTIVE reviews, i.e. each reviewer's individual reviews of papers, they need not be derived from some global true scores. This dataset is used by our subjective formulation, where instead of true scores and true ranking among all papers, we instead tries to recover the scores that reviewers would have given to papers if not for the quantization. 

The goal for estimators using this dataset is to recover a set of scores that are intuitive for the area chairs to interpretate, and incorporates the rankings as well. 
"""
import numpy as np
import sys
sys.path.append('..')
import os
from utils.utils_funcs import *
import itertools
from scipy import stats


class Dataset_sub():
    """
    This class provide data to the solvers.
    Difference with Dataset(): this one assumes subjective reviews from each reviewers
    """
    def __init__(self, num_item, configs=None):
        self.num_item = num_item
        self.configs = configs
        self.has_data = False
    
    def get_combined_data(self, num_scores, num_assigns, loadpath = None, savepath = None, model = 'Thurstone', data_dist = 'uniform', noise_std = 1.0):
        """
        num_assigns: The number of papers each reviewer gets. They will provide a full ranking among these papers, as well as quantized scores for each. 
        num_scores: The number of times each paper is reviewed. 
        model: the generative model. So far we only support Thurstone. 
        """
        # assert model.lower()=='thurstone'
        # assert self.num_item % num_assigns == 0
        if loadpath is None: 
            reviewer_to_paper = [] # num_reviewer x num_assigns
            reviewer_to_score = []
            paper_to_reviewer = np.full((self.num_item,num_scores), -1)
            paper_to_score = np.zeros((self.num_item, num_scores))
            if data_dist.lower()=='uniform':
                data_x = np.random.uniform(low = 1, high = 9.0, size = (self.num_item,))
            elif data_dist.lower()=='gaussian':
                data_x = np.random.normal(loc=5.0, scale = 2.0, size= (self.num_item,))
                np.clip(data_x, 0, 10, out=data_x)
            else:
                raise ValueError("Do not recognize data_dist = {}".format(data_dist))
            reviewer_index = 0
            for score_position in range(num_scores):
                perm = np.random.permutation(self.num_item)
                for paper_ids in np.split(perm, self.num_item/num_assigns):
                    scores = data_x[paper_ids]+np.random.normal(0,noise_std,num_assigns)
                    np.clip(scores, 0, 10, out=scores)
                    
                    ranked_ind = np.argsort(scores)[::-1] # indices that rank papers in paper_ids in decreasing order
                    reviewer_to_score.append(scores[ranked_ind])
                    reviewer_to_paper.append(paper_ids[ranked_ind])
                    
                    paper_to_score[paper_ids, np.full((num_assigns,), score_position)] = scores
                    paper_to_reviewer[paper_ids, np.full((num_assigns,), score_position)] = reviewer_index
                    reviewer_index += 1

            reviewer_to_paper = np.vstack(reviewer_to_paper)
            reviewer_to_score = np.vstack(reviewer_to_score)
            self.reviewer_to_paper = reviewer_to_paper
            self.reviewer_to_score = reviewer_to_score
            self.reviewer_to_rating = np.round(self.reviewer_to_score)
            self.paper_to_reviewer = paper_to_reviewer
            self.paper_to_score = paper_to_score
            self.paper_to_rating = np.round(self.paper_to_score)
            self.noise_std = noise_std
            self.data_x = data_x
            
            if savepath is not None:
                np.savez(savepath, reviewer_to_paper=reviewer_to_paper, reviewer_to_score = reviewer_to_score, paper_to_reviewer = paper_to_reviewer, paper_to_score = paper_to_score, noise_std = noise_std, data_x = data_x)
                print("saved data to %s" % savepath)

        else: 
            try: 
                data = np.load(loadpath)
                self.reviewer_to_paper = data['reviewer_to_paper']
                self.reviewer_to_score = data['reviewer_to_score']
                self.reviewer_to_rating = np.round(self.reviewer_to_score)
                self.paper_to_reviewer = data['paper_to_reviewer']
                self.paper_to_score = data['paper_to_score']
                self.paper_to_rating = np.round(self.paper_to_score)
                self.noise_std = data['noise_std']
                self.data_x = data['data_x']
            except:
                print("check this path: %s" % loadpath)
        self.has_data =True
    
    def partial_kendalltau_dist(self, test_y, regularize=True):
        """For all pairs (i,j) where i<j, check disagreement only if original score(i)!= original_score(j) """
        true_y = self.paper_to_score.flatten()
        errors = partial_kendalltau_dist_utils(true_y, test_y, regularize, False)
        return np.mean(errors)