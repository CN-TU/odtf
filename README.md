# ODTF: one-class decision tree fuzzifier

Designed/created by Félix Iglesias, TU Wien (Feb 2018)

Refined/optimize by Matthias Katzengruber, TU Wien (May 2018)

### Scripts:

- "odtfw.py" contains the one-class membership wrapper for decision trees.

- "memb_dist.py" calculates the distributions (with 10,50 and 90 quantiles/percentiles) of FNs, TNs, TPs and FPs.

###  Available data:

- The [knn] folder contains all data, configuration files and results for the experiments conducted in paper [1] related to the KDD-NSL dataset [2][3]. Data is already normalized, categorical removed and PCA transformed as explained in the paper.

- The [unsw] folder contains all data, configuration files and results for the experiments conducted in paper [1] related to the UNSW-NB15 dataset [4][5][6]. Data is already normalized, categorical removed and PCA transformed as explained in the paper.

###  Reproducing experiments:

Experiments from paper [1] can be reproduced with the following commands: 

**Warning: These operations will overwrite previous results stored in [out] subfolders!**

**Pre-step: Extract "test.csv" and "train.csv" files from "test.csv.zip" and "train.csv.zip" from the desired subfolder before running the scripts!**

- *(KDD dataset) Calculating normal performances and one-class membership scores*
```
> python odtfw.py kdd/config_num_norm_pca_test.txt
```
- *(KDD dataset) Showing membership score distributions of FPs, FNs, TPs and TNs*
```
> python memb_dist.py kdd/config_membership_test.txt
```
- *(KDD dataset) Calculating performances using membership scores as an additional feature*
```
> python odtfw.py kdd/config_num_norm_pca_with-memb_test.txt
```

- *(UNSW-NB15 dataset) Calculating normal performances and one-class membership scores*
```
> python odtfw.py unsw/config_num_norm_pca_test.txt
```
- *(UNSW-NB15 dataset) Showing membership score distributions of FPs, FNs, TPs and TNs*
```
> python memb_dist.py unsw/config_membership_test.txt
```
- *(UNSW-NB15 dataset) Calculating performances using membership scores as an additional feature*
```
> python odtfw.py unsw/config_num_norm_pca_with-memb_test.txt
```

To recalculate training data memberships, you can use the following commands for NSL-KDD and UNSW-NB15:
```
> python odtfw.py kdd/config_num_norm_pca_train.txt

> python odtfw.py unsw/config_num_norm_pca_train.txt
```
### References
- [1] publication pending
- [2] M. Tavallaee, E. Bagheri, W. Lu and A. A. Ghorbani, "A detailed analysis of the KDD CUP 99 data set," 2009 IEEE Symposium on Computational Intelligence for Security and Defense Applications, Ottawa, ON, 2009, pp. 1-6. URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5356528&isnumber=5356514
- [3] NSL-KDD dataset: http://www.unb.ca/cic/datasets/nsl.html
- [4] Moustafa, Nour, and Jill Slay. "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)."Military Communications and Information Systems Conference (MilCIS), 2015. IEEE, 2015.
- [5] Moustafa, Nour, and Jill Slay. "The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set." Information Security Journal: A Global Perspective (2016): 1-14.
- [6] UNSW-NB15 dataset: https://www.unsw.adfa.edu.au/australian-centre-for-cyber-security/cybersecurity/ADFA-NB15-Datasets/
