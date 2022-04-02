python src/exp_dequantize_scores_CrossValidation.py \
--n_item 60 \
--num_tests 20 \
--num_scores 4 \
--num_assigns 4 \
--sigma 0.5 \
--nu 0.05 \
--variance_loss \
--save

# python src/exp_dequantize_scores_iclr_CrossValidation.py \
# --num_tests 20 \
# --num_assigns 6 \
# --nu 0.05 \
# --variance_loss \
# --comp_constraint partial \
# --save