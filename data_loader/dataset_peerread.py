import sys,os,random,json,glob,operator,re
from collections import Counter
sys.path.insert(1,os.path.join(sys.path[0],'..'))
from utils.utils_funcs import *
from peerread_models.Review import Review
from peerread_models.Paper import Paper
from scipy import stats
# from peerread_models.ScienceParse import ScienceParse
# from peerread_models.ScienceParseReader import ScienceParseReader

import ipdb
import numpy as np


class Dataset_sub_iclr():
    """
    This class provide data to the solvers.
    Difference with Dataset(): this one assumes subjective reviews from each r eviewers
    """
    def __init__(self, reviewer_mode="random"):
        self.num_assigns= 0
        self.num_scores = 3 
        self.reviewer_mode = reviewer_mode
        self.reviewer_to_paper = [] # num_reviewer x num_assigns
        self.reviewer_to_rating = []
        self.paper_to_reviewer = []
        self.paper_to_rating = []
    
    def load_data(self, path="peerread_data/iclr_2017"):
        ################################
        # read reviews
        ################################
        papers = []
        list_of_scores = []
        lost_scores = 0
        total_scores = 0
        for split in ["train", "test", "dev"]:
            paper_json_dir = os.path.join(path, split, "reviews")
            print('Reading reviews from...',paper_json_dir)
            paper_json_filenames = sorted(glob.glob('{}/*.json'.format(paper_json_dir)))

            for paper_json_filename in paper_json_filenames:
                paper = Paper.from_json(paper_json_filename)
                scores = []
                if not paper: continue
                reviews = paper.get_reviews()
                comments = [] # screen out duplicate reviews
                for r in reviews:
                    if 'RECOMMENDATION' in r.__dict__ and r.__dict__['RECOMMENDATION'] is not None:
                        if 'COMMENTS' in r.__dict__ and r.__dict__['COMMENTS'] is not "":
                            if r.__dict__['COMMENTS'] not in comments:
                                scores.append(r.get_recommendation())
                                comments.append(r.get_comments())
                total_scores += len(scores)
                if len(scores)>self.num_scores:
                    lost_scores += len(scores) - self.num_scores
                    scores = scores[:self.num_scores]
                
                if len(scores)<self.num_scores:
                    raise ValueError("{} doesn't have enough score of {}, it only has {} scores".format(paper_json_filename, self.num_scores, len(scores)))
                
                list_of_scores.append(scores)
                papers.append(paper)
        print('Total number of papers',len(papers))
        print('Lost {} out of {} scores to ensure every paper have the same number of scores'.format(lost_scores, total_scores))
        return papers, list_of_scores


    def get_data(self, num_assigns, path="peerread_data/iclr_2017", comparison_constraint = "partial"):
        assert comparison_constraint in ["partial", "random", "strict"]
        self.num_assigns = num_assigns # start from 2
        papers, original_scores = self.load_data(path=path)
        if (len(papers)%2) == 1:
            papers = papers[:-1]
            original_scores = original_scores[:-1]
            print("Discard last paper for simplicity, need num of papers to be even")
        self.num_item = len(papers)
        
        self.original_scores = np.vstack(original_scores)
        """ Put into 5 bins """
        self.paper_to_rating = np.ceil(self.original_scores/2)

        """Further quantize"""
        # self.original_scores = self.paper_to_rating
        # self.paper_to_rating = np.ceil(self.original_scores/2)
        self.paper_to_reviewer = np.full((self.num_item,self.num_scores), -1)

        if comparison_constraint == "random":
            self.y_ranking = get_rank(self.original_scores.flatten()).reshape(self.original_scores.shape)
            if self.reviewer_mode=="random":
                reviewer_index = 0
                for score_position in range(self.num_scores):
                    perm = np.random.permutation(self.num_item)
                    for paper_ids in np.split(perm, self.num_item/num_assigns):
                        ranked_ind = np.argsort(self.y_ranking[paper_ids, np.full((num_assigns,), score_position)])
                        self.reviewer_to_rating.append(self.paper_to_rating[paper_ids[ranked_ind], np.full((num_assigns,), score_position)])
                        self.reviewer_to_paper.append(paper_ids[ranked_ind])
                        
                        self.paper_to_reviewer[paper_ids, np.full((num_assigns,), score_position)] = reviewer_index

                        reviewer_index += 1
                
                self.reviewer_to_paper = np.vstack(self.reviewer_to_paper)
                self.reviewer_to_rating = np.vstack(self.reviewer_to_rating)
        
        elif comparison_constraint=="partial":
            self.reviewer_to_list_of_groups = []
            if self.reviewer_mode=="random":
                reviewer_index = 0
                for score_position in range(self.num_scores):
                    perm = np.random.permutation(self.num_item)
                    for paper_ids in np.split(perm, self.num_item/num_assigns):
                        z = self.original_scores[paper_ids,np.full((num_assigns,), score_position)]
                        ranked_ind = np.argsort(z)[::-1]
                        self.reviewer_to_rating.append(self.paper_to_rating[paper_ids[ranked_ind], np.full((num_assigns,), score_position)])
                        list_of_groups = [np.where(z[ranked_ind]==v)[0] for v in sorted(np.unique(z),reverse=True)]
                        self.reviewer_to_list_of_groups.append(list_of_groups)# CHANGED PART
                        self.reviewer_to_paper.append(paper_ids[ranked_ind]) 
                        self.paper_to_reviewer[paper_ids, np.full((num_assigns,), score_position)] = reviewer_index
                        reviewer_index += 1
                self.reviewer_to_paper = np.vstack(self.reviewer_to_paper)
                self.reviewer_to_rating = np.vstack(self.reviewer_to_rating)
        
        else: 
            pass
    
    def partial_kendalltau_dist(self, test_y, regularize=True):
        """For all pairs (i,j) where i<j, check disagreement only if original score(i)!= original_score(j) """
        true_y = self.original_scores.flatten()
        errors = partial_kendalltau_dist_utils(true_y, test_y, regularize, False) 
        return np.mean(errors)
    
    