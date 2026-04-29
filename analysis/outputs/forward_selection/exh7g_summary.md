# 7-var exhaustive search — full metric grid

Candidates: ['cross_domain_attack', 'conf_gap_change', 'var_conf_pro', 'H_final', 'delta_H', 'semantic_coherence', 'prediction_volatility']


Metric slots evaluated: 3 metrics × 4 k% = **12 slots**
(Precision / Recall / DOR @ k% for k ∈ [5, 10, 15, 20])


Cohort: n=7,086, Y rate 12.49%. 6 models × 5 seeds = 30 fits per subset.


## Consensus winners across 12 slots

Subsets ranked by appearances in top-1 across all slots (higher = robust winner across metric choices).

| subset_label                                                                              |   in_top1_count |   in_top3_count |   in_top5_count |   subset_size |
|:------------------------------------------------------------------------------------------|----------------:|----------------:|----------------:|--------------:|
| cross_domain_attack+semantic_coherence+prediction_volatility                              |               6 |               6 |               6 |             3 |
| cross_domain_attack+semantic_coherence                                                    |               3 |               4 |               6 |             2 |
| delta_H+semantic_coherence                                                                |               3 |               3 |               3 |             2 |
| cross_domain_attack+H_final+semantic_coherence                                            |               0 |               3 |               3 |             3 |
| cross_domain_attack+conf_gap_change+prediction_volatility                                 |               0 |               3 |               3 |             3 |
| cross_domain_attack+conf_gap_change+var_conf_pro+semantic_coherence+prediction_volatility |               0 |               3 |               3 |             5 |
| cross_domain_attack+H_final+delta_H+semantic_coherence                                    |               0 |               3 |               3 |             4 |
| var_conf_pro+H_final+semantic_coherence                                                   |               0 |               3 |               3 |             3 |
| cross_domain_attack+H_final+semantic_coherence+prediction_volatility                      |               0 |               3 |               3 |             4 |
| semantic_coherence                                                                        |               0 |               2 |               3 |             1 |
| var_conf_pro                                                                              |               0 |               2 |               2 |             1 |
| H_final+delta_H+semantic_coherence                                                        |               0 |               1 |               1 |             3 |
| var_conf_pro+delta_H+semantic_coherence                                                   |               0 |               0 |               5 |             3 |
| conf_gap_change+H_final+semantic_coherence+prediction_volatility                          |               0 |               0 |               3 |             4 |
| cross_domain_attack+conf_gap_change+delta_H                                               |               0 |               0 |               3 |             3 |


## Per-slot top-3 winners


### Precision @ top-5%  (BIBLIO = 0.39343)

|   subset_size | subset_label                                   |   Precision_mean |   Precision_std |   delta_vs_BIBLIO |
|--------------:|:-----------------------------------------------|-----------------:|----------------:|------------------:|
|             2 | cross_domain_attack+semantic_coherence         |          0.40235 |         0.07376 |           0.00892 |
|             3 | cross_domain_attack+H_final+semantic_coherence |          0.40141 |         0.07605 |           0.00798 |
|             3 | var_conf_pro+H_final+semantic_coherence        |          0.40094 |         0.07835 |           0.00751 |

### Recall @ top-5%  (BIBLIO = 0.15782)

|   subset_size | subset_label                                   |   Recall_mean |   Recall_std |   delta_vs_BIBLIO |
|--------------:|:-----------------------------------------------|--------------:|-------------:|------------------:|
|             2 | cross_domain_attack+semantic_coherence         |       0.16139 |      0.02959 |           0.00357 |
|             3 | cross_domain_attack+H_final+semantic_coherence |       0.16102 |      0.03051 |           0.0032  |
|             3 | var_conf_pro+H_final+semantic_coherence        |       0.16083 |      0.03143 |           0.00301 |

### DOR @ top-5%  (BIBLIO = 5.49501)

|   subset_size | subset_label                                   |   DOR_mean |   DOR_std |   delta_vs_BIBLIO |
|--------------:|:-----------------------------------------------|-----------:|----------:|------------------:|
|             2 | cross_domain_attack+semantic_coherence         |    5.733   |   2.03807 |           0.23799 |
|             3 | var_conf_pro+H_final+semantic_coherence        |    5.7307  |   2.15998 |           0.23569 |
|             3 | cross_domain_attack+H_final+semantic_coherence |    5.72496 |   2.09386 |           0.22995 |

### Precision @ top-10%  (BIBLIO = 0.33052)

|   subset_size | subset_label                                                         |   Precision_mean |   Precision_std |   delta_vs_BIBLIO |
|--------------:|:---------------------------------------------------------------------|-----------------:|----------------:|------------------:|
|             3 | cross_domain_attack+semantic_coherence+prediction_volatility         |          0.34272 |         0.05229 |           0.0122  |
|             4 | cross_domain_attack+H_final+semantic_coherence+prediction_volatility |          0.33991 |         0.05231 |           0.00939 |
|             4 | cross_domain_attack+H_final+delta_H+semantic_coherence               |          0.33709 |         0.04965 |           0.00657 |

### Recall @ top-10%  (BIBLIO = 0.26516)

|   subset_size | subset_label                                                         |   Recall_mean |   Recall_std |   delta_vs_BIBLIO |
|--------------:|:---------------------------------------------------------------------|--------------:|-------------:|------------------:|
|             3 | cross_domain_attack+semantic_coherence+prediction_volatility         |       0.27495 |      0.04195 |           0.00979 |
|             4 | cross_domain_attack+H_final+semantic_coherence+prediction_volatility |       0.27269 |      0.04196 |           0.00753 |
|             4 | cross_domain_attack+H_final+delta_H+semantic_coherence               |       0.27043 |      0.03983 |           0.00527 |

### DOR @ top-10%  (BIBLIO = 4.49632)

|   subset_size | subset_label                                                         |   DOR_mean |   DOR_std |   delta_vs_BIBLIO |
|--------------:|:---------------------------------------------------------------------|-----------:|----------:|------------------:|
|             3 | cross_domain_attack+semantic_coherence+prediction_volatility         |    4.83608 |   1.39997 |           0.33976 |
|             4 | cross_domain_attack+H_final+semantic_coherence+prediction_volatility |    4.76187 |   1.40046 |           0.26555 |
|             4 | cross_domain_attack+H_final+delta_H+semantic_coherence               |    4.67111 |   1.31337 |           0.17479 |

### Precision @ top-15%  (BIBLIO = 0.29421)

|   subset_size | subset_label                                                                              |   Precision_mean |   Precision_std |   delta_vs_BIBLIO |
|--------------:|:------------------------------------------------------------------------------------------|-----------------:|----------------:|------------------:|
|             3 | cross_domain_attack+semantic_coherence+prediction_volatility                              |          0.30469 |         0.0403  |           0.01048 |
|             5 | cross_domain_attack+conf_gap_change+var_conf_pro+semantic_coherence+prediction_volatility |          0.30329 |         0.0395  |           0.00908 |
|             1 | var_conf_pro                                                                              |          0.30313 |         0.03706 |           0.00892 |

### Recall @ top-15%  (BIBLIO = 0.35405)

|   subset_size | subset_label                                                                              |   Recall_mean |   Recall_std |   delta_vs_BIBLIO |
|--------------:|:------------------------------------------------------------------------------------------|--------------:|-------------:|------------------:|
|             3 | cross_domain_attack+semantic_coherence+prediction_volatility                              |       0.36667 |      0.0485  |           0.01262 |
|             5 | cross_domain_attack+conf_gap_change+var_conf_pro+semantic_coherence+prediction_volatility |       0.36497 |      0.04754 |           0.01092 |
|             1 | var_conf_pro                                                                              |       0.36478 |      0.0446  |           0.01073 |

### DOR @ top-15%  (BIBLIO = 4.10900)

|   subset_size | subset_label                                                                              |   DOR_mean |   DOR_std |   delta_vs_BIBLIO |
|--------------:|:------------------------------------------------------------------------------------------|-----------:|----------:|------------------:|
|             3 | cross_domain_attack+semantic_coherence+prediction_volatility                              |    4.4115  |   1.18791 |           0.3025  |
|             5 | cross_domain_attack+conf_gap_change+var_conf_pro+semantic_coherence+prediction_volatility |    4.35941 |   1.10014 |           0.25041 |
|             2 | cross_domain_attack+semantic_coherence                                                    |    4.35808 |   1.19788 |           0.24908 |

### Precision @ top-20%  (BIBLIO = 0.27430)

|   subset_size | subset_label                                              |   Precision_mean |   Precision_std |   delta_vs_BIBLIO |
|--------------:|:----------------------------------------------------------|-----------------:|----------------:|------------------:|
|             2 | delta_H+semantic_coherence                                |          0.2804  |         0.03157 |           0.0061  |
|             1 | semantic_coherence                                        |          0.27852 |         0.02687 |           0.00422 |
|             3 | cross_domain_attack+conf_gap_change+prediction_volatility |          0.2784  |         0.03409 |           0.0041  |

### Recall @ top-20%  (BIBLIO = 0.44011)

|   subset_size | subset_label                                              |   Recall_mean |   Recall_std |   delta_vs_BIBLIO |
|--------------:|:----------------------------------------------------------|--------------:|-------------:|------------------:|
|             2 | delta_H+semantic_coherence                                |       0.44991 |      0.05065 |           0.0098  |
|             1 | semantic_coherence                                        |       0.44689 |      0.04311 |           0.00678 |
|             3 | cross_domain_attack+conf_gap_change+prediction_volatility |       0.4467  |      0.05469 |           0.00659 |

### DOR @ top-20%  (BIBLIO = 4.06028)

|   subset_size | subset_label                                              |   DOR_mean |   DOR_std |   delta_vs_BIBLIO |
|--------------:|:----------------------------------------------------------|-----------:|----------:|------------------:|
|             2 | delta_H+semantic_coherence                                |    4.26813 |   1.03593 |           0.20785 |
|             3 | cross_domain_attack+conf_gap_change+prediction_volatility |    4.21597 |   1.06847 |           0.15569 |
|             3 | H_final+delta_H+semantic_coherence                        |    4.19208 |   1.00563 |           0.1318  |


## AUROC ranking (top 10) for reference

|   subset_size | subset_label                                                                              |   AUROC_mean |   delta_AUROC |
|--------------:|:------------------------------------------------------------------------------------------|-------------:|--------------:|
|             3 | H_final+delta_H+semantic_coherence                                                        |      0.72445 |       0.00999 |
|             4 | var_conf_pro+H_final+delta_H+semantic_coherence                                           |      0.72403 |       0.00957 |
|             4 | cross_domain_attack+H_final+delta_H+semantic_coherence                                    |      0.72393 |       0.00947 |
|             2 | conf_gap_change+var_conf_pro                                                              |      0.72368 |       0.00922 |
|             3 | conf_gap_change+var_conf_pro+semantic_coherence                                           |      0.72359 |       0.00913 |
|             5 | cross_domain_attack+conf_gap_change+H_final+delta_H+semantic_coherence                    |      0.72311 |       0.00865 |
|             3 | cross_domain_attack+delta_H+semantic_coherence                                            |      0.7228  |       0.00834 |
|             4 | conf_gap_change+var_conf_pro+semantic_coherence+prediction_volatility                     |      0.72263 |       0.00817 |
|             4 | var_conf_pro+delta_H+semantic_coherence+prediction_volatility                             |      0.72262 |       0.00816 |
|             5 | cross_domain_attack+conf_gap_change+var_conf_pro+semantic_coherence+prediction_volatility |      0.72255 |       0.00809 |