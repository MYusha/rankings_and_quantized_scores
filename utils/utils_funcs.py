import numpy as np
import math
# import torch
import matplotlib.pyplot as plt

# torch.set_default_tensor_type(
#             torch.cuda.FloatTensor if torch.cuda.is_available()
#             else torch.FloatTensor)

# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
# Tensor = FloatTensor
# print("Using", torch.cuda.device_count(), "GPUs")

def to_np(tensor):
	"""converts torch tensor to numpy"""
	data = tensor.data.cpu().numpy()
	return data

def plot_mat(valmat,k_l, k_r,b_l,b_r):
	'''Plots'''
	xm, xn = valmat.shape
	fig, ax = plt.subplots()
	cmap=plt.cm.OrRd
	cmap.set_under(color='gray')
	plt.imshow(valmat, cmap=cmap)
	for i in range(xn): # i = num_bnins? / x-axis
		for j in range(xm): # j = k
			c = valmat[j,i]
			ax.text(i, j, "{:.3f}".format(c*10), va='center', ha='center')

	# plt.gca().set_xticklabels(np.arange(1, 6, 1))
	# plt.gca().set_yticklabels(np.arange(1, 5, 1))
	ax.set_xticks(np.arange(xn))
	ax.set_yticks(np.arange(xm))
	# ... and label them with the respective list entries
	ax.set_xticklabels(np.arange(b_l,b_r+1,1))
	ax.set_yticklabels(np.arange(k_l,k_r+1,1))
	ax.tick_params(top=True, bottom=False,
					   labeltop=True, labelbottom=False)
	plt.xlabel('b value')
	plt.ylabel('k value')
	plt.title('Increment in regularized Kendall-tau distance (10^-1)',y=-0.01)
	plt.show()

def get_index(x, tie = "random", order = "descend"):
	"""
	return index that sort the elements in descending order, break ties randomly
	"""
	x = x.flatten()
	if tie=="random":
		shuffle_ = np.random.uniform(size=len(x)) # (d, )
		idx = np.lexsort((shuffle_,x)) # sort by a, then by b
		if order=="descend":
			idx= idx[::-1] # descending order
	else:
		raise ValueError("other break tie method not supported now")
	return idx

def get_rank(x, tie = "random", order = "descend"):
	"""
	Input x: numpy array (d,)
	Output rank: numpy array(d,) is the descending rank corresponding to each item in x
	break ties randomly
	"""
	if tie == "random":
		idx = get_index(x, tie, order)
		# temp = np.argsort(x)[::-1] # indices for descending order
		ranks = np.zeros_like(x, dtype=np.int)
		ranks[idx] = np.arange(len(x))
	else:
		raise ValueError("other method not supported now")
	return ranks

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def kendalltau_dist(true_a, test_b, regularize=True, use_rank = True):
	n = len(true_a)
	assert n==len(test_b)
	if use_rank: 
		""" If vector true_a and test_b are ranks. Assumes no ties."""
		diff_a = np.expand_dims(true_a, axis=1) - np.expand_dims(true_a, axis=0) # element {i, j} = a(i) - a(j)
		diff_b = np.expand_dims(test_b, axis=1) - np.expand_dims(test_b, axis=0)
		idxs_a = np.where(diff_a > 0) # where ra(i)>ra(j)
		tau = len(np.where(diff_b[idxs_a] < 0)[0]) # i, j where r(i)> ra(j) but rb(i)<rb(j)
	else:
		""" If vectors are scores(that induce ranks). Allows ties and treats them as 0.5 error per tie"""
		diff_a = np.expand_dims(true_a, axis=1) - np.expand_dims(true_a, axis=0) # element {i, j} = a(i) - a(j)
		idxs_a = np.where(diff_a > 0) # where ra(i)>ra(j)
		diff_b = np.expand_dims(test_b, axis=1) - np.expand_dims(test_b, axis=0)
		tau = len(np.where(diff_b[idxs_a] < 0)[0]) # i, j where ra(i)> ra(j) but rb(i)<rb(j)
		tau_ties = len(np.where(diff_b[idxs_a] == 0)[0]) # i, j where ra(i)> ra(j) but rb(i)=rb(j) (ties. )
		tau += tau_ties * 0.5
	if regularize:
		tau = 2*tau/(n*(n-1))
	return tau

def partial_kendalltau_dist_utils(true_y, test_y, regularize=True, random_ties=True):
	"""For all pairs (i,j) where i<j, check disagreement only if original score_(i)!= original_score(j) """
	# print("calling utils partial kt dist")
	assert len(test_y)==len(true_y)
	diff_true_y = np.expand_dims(true_y, axis=1) - np.expand_dims(true_y, axis=0) # element {i, j} = true_y(i) - true_y(j)
	true_i_beats_j_flat_idx = np.where(diff_true_y.flatten()>0)
	num_ij = len(true_i_beats_j_flat_idx[0]) #|{i, j, s.t. y_i > y_j}|
	# print("num_ij = {}".format(num_ij))
	if random_ties:
		# break ties in test_y randomly, need the master algorithm to run multiples times to approximate the expected KT distance
		test_y += np.random.uniform(low=-1e-6, high=1e-6, size=test_y.shape) # only to break ties
		diff_test_y = np.expand_dims(test_y, axis=1) - np.expand_dims(test_y, axis=0)
		tau_1 = len(np.where(diff_test_y.flatten()[true_i_beats_j_flat_idx]<0)[0])

	else:
		# treat ties as 0.5 error, directly calculate expected KT distance. 
		diff_test_y = np.expand_dims(test_y, axis=1) - np.expand_dims(test_y, axis=0)
		tau_1 = len(np.where(diff_test_y.flatten()[true_i_beats_j_flat_idx]<0)[0])
		""" calculate number of ties"""
		tau_1_ties = 0.5* len(np.where(diff_test_y.flatten()[true_i_beats_j_flat_idx]==0)[0])		
		tau_1 = tau_1 + tau_1_ties
	if regularize:
		tau_1 = tau_1/num_ij
	return tau_1
