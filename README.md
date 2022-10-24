# ew_rhals
This repository contains the necessary functions to run the RHALS, HALS, and ALS algorithms described in "Positive Matrix Factorization of Large Aerosol Mass Spectrometry Datasets Using Error-Weighted Randomized Hierarchical Alternating Least Squares"

randomized_nnmf.m contains the Randomized HALS algorithm, while exactnnmf.m contains the deterministic HALS algorithm. The other functions compressed_mu.m, 
reg_als.m, and mult_update.m are the other algorithms tested in the paper. All functions rely oninitializefactors.m, initializewh.m, and LOCAL_rsvd.m.

exactnnmf.m relies on updateWH.m for each iteration. To look at pulling, use updateWH_pull.m.
randomized_nnmf.m relies only on updateWHrandom.m for each iteration.

