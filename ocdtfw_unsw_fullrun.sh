#! /bin/bash

printf "\n --------------------- ODTF training/training (UNSW dataset) --------------------- \n"
python ocdtfw.py unsw/config_num_norm_pca_train.txt
printf "\n --------------------- ODTF training/test (UNSW dataset) --------------------- \n"
python ocdtfw.py unsw/config_num_norm_pca_test.txt

printf "\n --------------------- Add memberships to datasets (UNSW dataset) --------------------- \n"
python addmembershipscore.py unsw/config_add_memb_train.txt
python addmembershipscore.py unsw/config_add_memb_test.txt

printf "\n --------------------- ODTF training/test (UNSW dataset with Memberships) --------------------- \n"
python ocdtfw.py unsw/config_num_norm_pca_with-memb_test.txt

printf "\n --------------------- Extract Membership quartiles --------------------- \n"
python memb_dist.py unsw/config_membership_test.txt
