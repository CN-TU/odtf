# ODTF: one-class decision tree fuzzifier

### Scripts:

- "ocdtfw.py" contains the one-class membership wrapper for decision trees.

- "memb_dist.py" calculates the distributions (with 10,50 and 90 quantiles/percentiles) of FNs, TNs, TPs and FPs.

###  Available data:

- The [knn] folder contains all data, configuration files and results for the experiments conducted in paper [1] related to the KDD-NSL dataset. Data is already normalized, categorical removed and PCA transformed as explained in the paper.

- The [unsw] folder contains all data, configuration files and results for the experiments conducted in paper [1] related to the UNSW-NB15 dataset. Data is already normalized, categorical removed and PCA transformed as explained in the paper.

###  Reproducing experiments:

Experiments from paper [1] can be reproduced with the following commands (Warning!: These operations will overwrite previous results stored in [out] subfolders):

- *(KDD dataset) Calculating normal performances and one-class membership scores*
```
> python ocdtfw.py kdd/config_num_norm_pca_test.txt
```
- *(KDD dataset) Showing membership score distributions of FPs, FNs, TPs and TNs*
```
> python memb_dist.py kdd/config_membership_test.txt
```
- *(KDD dataset) Calculating performances using membership scores as an additional feature*
```
> python ocdtfw.py kdd/config_num_norm_pca_with-memb_test.txt
```

- *(UNSW-NB15 dataset) Calculating normal performances and one-class membership scores*
```
> python ocdtfw.py unsw/config_num_norm_pca_test.txt
```
- *(UNSW-NB15 dataset) Showing membership score distributions of FPs, FNs, TPs and TNs*
```
> python memb_dist.py unsw/config_membership_test.txt
```
- *(UNSW-NB15 dataset) Calculating performances using membership scores as an additional feature*
```
> python ocdtfw.py unsw/config_num_norm_pca_with-memb_test.txt
```

To recalculate training data memberships, you can use the following commands for NSL-KDD and UNSW-NB15:
```
> python ocdtfw.py kdd/config_num_norm_pca_train.txt

> python ocdtfw.py unsw/config_num_norm_pca_train.txt
```
### References
[1] publication pending
