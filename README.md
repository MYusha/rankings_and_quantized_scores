# Rankings and quantized ratings
This is the code for paper "Integrating Rankings into Quantized Scores in Peer Review".

## Citation
Liu, Yusha, Yichong Xu, Nihar B. Shah, and Aarti Singh. <br/>
[Integrating Rankings into Quantized Scores in Peer Review](https://arxiv.org/abs/2204.03505). <br/>
arXiv preprint arXiv:2204.03505 (2022).<br/>

## Requirements
Run with the following: 
* python 3.6
* cvxpy 1.1.12
* cvxopt 1.2.6

## Dataset Preperation
For the real-world data (Section 5.4), we use the ICLR 2017 dataset from [PeerRead](https://github.com/allenai/PeerRead) dataset.  Please see the citations therein. 

Put the ICLR 2017 data (in their original ``train, dev, test`` folders) from the PeerRead repository into ``peerread_data`` folder. Put the scripts from https://github.com/allenai/PeerRead/tree/master/code/models in the ``peerread_models`` folder. 

## Experiments
**Dequantize scores:** The scripts to conduct experiments with the proposed algorithm and baseline algorithms can be found in: \
``src/exp_dequantize_scores_CrossValidation.py`` for the synthetic data
``src/exp_dequantize_scores_iclr_CrossValidation.py`` for the ICLR 2017 data.\
<br>

For an example on how to run them, see the script ``run_sc.sh``. Here are some explanations of input parameters (see Section 5 of the paper for the corresponding definitions):  
``--n_item``: Number of papers in the synthetic dataset.<br>
``--num_tests``: Number of trials.<br>
``--num_scores``: Number of reviews received by each paper.<br>
``--num_assigns``: Number of papers assigned to each reviewer. <br>
``--sigma``: Value of the noise standard deviation in Thurstone model. <br>
``--nu``: Value of the small constant in the constraints in proposed algorithm. <br>
``--save``: Whether to save results. <br>


The results, which are the dequantized scores, and other information including the information and data of the generated datasets will be saved in the ``results`` folder. 

**Visualizing results:** The script to visualize the results is ``visualization/viz_all.py``.
