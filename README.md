# Rankings and quantized ratings
This is the code for the paper "Integrating Rankings into Quantized Scores in Peer Review". The citation information will soon be updated. 

## Requirements
<!-- Run with the following:  -->
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
For an example on how to run them, see the script ``run_sc.sh``. 

The generated results are the dequantized scores, as well as other information including the generated datasets and values of their parameters. The results will be saved in the ``results`` folder. 

**Visualizing results:** The script to visualize the results is ``visualization/viz_all.py``.

<!-- ## Citations -->
