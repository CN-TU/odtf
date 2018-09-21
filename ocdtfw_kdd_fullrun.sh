#! /bin/bash

printf "\n --------------------- ODTF training/training (KDD dataset) --------------------- \n"
python ocdtfw.py kdd/config_num_norm_pca_train.txt
printf "\n --------------------- ODTF training/test (KDD dataset) --------------------- \n"
python ocdtfw.py kdd/config_num_norm_pca_test.txt

printf "\n --------------------- Add memberships to datasets (KDD dataset) --------------------- \n"
python addmembershipscore.py kdd/config_add_memb_train.txt
python addmembershipscore.py kdd/config_add_memb_test.txt

printf "\n --------------------- ODTF training/test (KDD dataset with Memberships) --------------------- \n"
python ocdtfw.py kdd/config_num_norm_pca_with-memb_test.txt

printf "\n --------------------- Extract Membership quartiles --------------------- \n"
python memb_dist.py kdd/config_membership_test.txt
