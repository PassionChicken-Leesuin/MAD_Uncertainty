# 6-var exhaustive search × 20 seeds × 6 models

**Candidates (6)**: cross_domain_attack, conf_gap_change, var_conf_pro, H_final, delta_H, semantic_coherence


Search: 2^6 - 1 = 63 non-empty + BIBLIO. Total fits: 7680.


Metric slots: 3 metrics × 4 k% = **12 slots** (Precision/Recall/DOR @ k% with k ∈ [5, 10, 15, 20])


Cohort: n=7,086, Y rate 12.49%.


## Consensus winners across 12 slots

| subset_label                                                |   subset_size |   in_top1_count |   in_top3_count |   in_top5_count |
|:------------------------------------------------------------|--------------:|----------------:|----------------:|----------------:|
| cross_domain_attack+semantic_coherence                      |             2 |               6 |               9 |              12 |
| var_conf_pro+semantic_coherence                             |             2 |               4 |               8 |               9 |
| BIBLIO_ONLY                                                 |             0 |               2 |               6 |               6 |
| cross_domain_attack+delta_H+semantic_coherence              |             3 |               0 |               6 |               8 |
| H_final                                                     |             1 |               0 |               3 |               3 |
| cross_domain_attack+var_conf_pro+H_final+semantic_coherence |             4 |               0 |               2 |               3 |
| cross_domain_attack+H_final+semantic_coherence              |             3 |               0 |               1 |               3 |
| cross_domain_attack+var_conf_pro+semantic_coherence         |             3 |               0 |               1 |               3 |
| cross_domain_attack                                         |             1 |               0 |               0 |               6 |
| delta_H+semantic_coherence                                  |             2 |               0 |               0 |               5 |
| cross_domain_attack+conf_gap_change+semantic_coherence      |             3 |               0 |               0 |               1 |
| var_conf_pro                                                |             1 |               0 |               0 |               1 |


## Per-slot top-3


### Precision @ top-5%  (BIBLIO = 0.39507)

|   subset_size | subset_label                    |   Precision_mean |   Precision_std |   delta_vs_BIBLIO |
|--------------:|:--------------------------------|-----------------:|----------------:|------------------:|
|             0 | BIBLIO_ONLY                     |          0.39507 |         0.07218 |           0       |
|             2 | var_conf_pro+semantic_coherence |          0.39413 |         0.07579 |          -0.00094 |
|             1 | H_final                         |          0.39272 |         0.07091 |          -0.00235 |

### Recall @ top-5%  (BIBLIO = 0.15847)

|   subset_size | subset_label                    |   Recall_mean |   Recall_std |   delta_vs_BIBLIO |
|--------------:|:--------------------------------|--------------:|-------------:|------------------:|
|             0 | BIBLIO_ONLY                     |       0.15847 |      0.02896 |           0       |
|             2 | var_conf_pro+semantic_coherence |       0.1581  |      0.0304  |          -0.00037 |
|             1 | H_final                         |       0.15753 |      0.02844 |          -0.00094 |

### DOR @ top-5%  (BIBLIO = 5.53090)

|   subset_size | subset_label                    |   DOR_mean |   DOR_std |   delta_vs_BIBLIO |
|--------------:|:--------------------------------|-----------:|----------:|------------------:|
|             2 | var_conf_pro+semantic_coherence |     5.5341 |   1.99089 |            0.0032 |
|             0 | BIBLIO_ONLY                     |     5.5309 |   1.8927  |            0      |
|             1 | H_final                         |     5.4624 |   1.85618 |           -0.0685 |

### Precision @ top-10%  (BIBLIO = 0.33239)

|   subset_size | subset_label                           |   Precision_mean |   Precision_std |   delta_vs_BIBLIO |
|--------------:|:---------------------------------------|-----------------:|----------------:|------------------:|
|             2 | var_conf_pro+semantic_coherence        |          0.33363 |         0.04716 |           0.00124 |
|             2 | cross_domain_attack+semantic_coherence |          0.33357 |         0.04606 |           0.00118 |
|             0 | BIBLIO_ONLY                            |          0.33239 |         0.0489  |           0       |

### Recall @ top-10%  (BIBLIO = 0.26667)

|   subset_size | subset_label                           |   Recall_mean |   Recall_std |   delta_vs_BIBLIO |
|--------------:|:---------------------------------------|--------------:|-------------:|------------------:|
|             2 | var_conf_pro+semantic_coherence        |       0.26766 |      0.03783 |           0.00099 |
|             2 | cross_domain_attack+semantic_coherence |       0.26761 |      0.03695 |           0.00094 |
|             0 | BIBLIO_ONLY                            |       0.26667 |      0.03923 |           0       |

### DOR @ top-10%  (BIBLIO = 4.54092)

|   subset_size | subset_label                           |   DOR_mean |   DOR_std |   delta_vs_BIBLIO |
|--------------:|:---------------------------------------|-----------:|----------:|------------------:|
|             2 | var_conf_pro+semantic_coherence        |    4.56518 |   1.17918 |           0.02426 |
|             2 | cross_domain_attack+semantic_coherence |    4.5603  |   1.18322 |           0.01938 |
|             0 | BIBLIO_ONLY                            |    4.54092 |   1.19485 |           0       |

### Precision @ top-15%  (BIBLIO = 0.29597)

|   subset_size | subset_label                                   |   Precision_mean |   Precision_std |   delta_vs_BIBLIO |
|--------------:|:-----------------------------------------------|-----------------:|----------------:|------------------:|
|             2 | cross_domain_attack+semantic_coherence         |          0.30192 |         0.03725 |           0.00595 |
|             2 | var_conf_pro+semantic_coherence                |          0.29992 |         0.03496 |           0.00395 |
|             3 | cross_domain_attack+delta_H+semantic_coherence |          0.2998  |         0.03697 |           0.00383 |

### Recall @ top-15%  (BIBLIO = 0.35617)

|   subset_size | subset_label                                   |   Recall_mean |   Recall_std |   delta_vs_BIBLIO |
|--------------:|:-----------------------------------------------|--------------:|-------------:|------------------:|
|             2 | cross_domain_attack+semantic_coherence         |       0.36332 |      0.04483 |           0.00715 |
|             2 | var_conf_pro+semantic_coherence                |       0.36092 |      0.04207 |           0.00475 |
|             3 | cross_domain_attack+delta_H+semantic_coherence |       0.36078 |      0.04449 |           0.00461 |

### DOR @ top-15%  (BIBLIO = 4.13380)

|   subset_size | subset_label                                        |   DOR_mean |   DOR_std |   delta_vs_BIBLIO |
|--------------:|:----------------------------------------------------|-----------:|----------:|------------------:|
|             2 | cross_domain_attack+semantic_coherence              |    4.30943 |   1.04603 |           0.17563 |
|             3 | cross_domain_attack+delta_H+semantic_coherence      |    4.24615 |   1.0251  |           0.11235 |
|             3 | cross_domain_attack+var_conf_pro+semantic_coherence |    4.23787 |   1.02345 |           0.10407 |

### Precision @ top-20%  (BIBLIO = 0.27236)

|   subset_size | subset_label                                                |   Precision_mean |   Precision_std |   delta_vs_BIBLIO |
|--------------:|:------------------------------------------------------------|-----------------:|----------------:|------------------:|
|             2 | cross_domain_attack+semantic_coherence                      |          0.27732 |         0.02828 |           0.00496 |
|             3 | cross_domain_attack+delta_H+semantic_coherence              |          0.2767  |         0.02828 |           0.00434 |
|             4 | cross_domain_attack+var_conf_pro+H_final+semantic_coherence |          0.27488 |         0.02866 |           0.00252 |

### Recall @ top-20%  (BIBLIO = 0.43701)

|   subset_size | subset_label                                                |   Recall_mean |   Recall_std |   delta_vs_BIBLIO |
|--------------:|:------------------------------------------------------------|--------------:|-------------:|------------------:|
|             2 | cross_domain_attack+semantic_coherence                      |       0.44496 |      0.04538 |           0.00795 |
|             3 | cross_domain_attack+delta_H+semantic_coherence              |       0.44397 |      0.04537 |           0.00696 |
|             4 | cross_domain_attack+var_conf_pro+H_final+semantic_coherence |       0.44105 |      0.04599 |           0.00404 |

### DOR @ top-20%  (BIBLIO = 3.98858)

|   subset_size | subset_label                                   |   DOR_mean |   DOR_std |   delta_vs_BIBLIO |
|--------------:|:-----------------------------------------------|-----------:|----------:|------------------:|
|             2 | cross_domain_attack+semantic_coherence         |    4.14234 |   0.91792 |           0.15376 |
|             3 | cross_domain_attack+delta_H+semantic_coherence |    4.11935 |   0.8886  |           0.13077 |
|             3 | cross_domain_attack+H_final+semantic_coherence |    4.06441 |   0.91781 |           0.07583 |


## AUROC top 10 (for reference)

|   subset_size | subset_label                                                           |   AUROC_mean |   AUROC_std |   delta_AUROC |
|--------------:|:-----------------------------------------------------------------------|-------------:|------------:|--------------:|
|             4 | cross_domain_attack+H_final+delta_H+semantic_coherence                 |      0.71976 |     0.03386 |       0.00517 |
|             5 | cross_domain_attack+conf_gap_change+H_final+delta_H+semantic_coherence |      0.71919 |     0.03539 |       0.0046  |
|             4 | var_conf_pro+H_final+delta_H+semantic_coherence                        |      0.71911 |     0.03477 |       0.00452 |
|             2 | var_conf_pro+semantic_coherence                                        |      0.71904 |     0.03456 |       0.00445 |
|             3 | cross_domain_attack+delta_H+semantic_coherence                         |      0.71896 |     0.03543 |       0.00437 |
|             3 | var_conf_pro+H_final+delta_H                                           |      0.71873 |     0.03666 |       0.00414 |
|             3 | var_conf_pro+H_final+semantic_coherence                                |      0.71847 |     0.03609 |       0.00388 |
|             3 | conf_gap_change+var_conf_pro+H_final                                   |      0.71846 |     0.03744 |       0.00387 |
|             3 | cross_domain_attack+H_final+semantic_coherence                         |      0.71836 |     0.0365  |       0.00377 |
|             3 | conf_gap_change+var_conf_pro+semantic_coherence                        |      0.71833 |     0.03635 |       0.00374 |