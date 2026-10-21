# CR-WBAN contextual-bandit simulator

Simulator, experiment scripts and result files for the paper
"A Contextual Bandit Approach for Adaptive Channel Selection in Cognitive Radio-Enabled Wireless Body Area Networks"
(H. Hussein, N. Tawfik). Repository DOI: **[to be inserted after upload]**. License: **[authors to choose, e.g. MIT]**.

## Quick start
```
pip install -r requirements.txt
./run_all.sh            # about 10 minutes, one CPU core
```
All results use 25 runs with seeds 1000-1024. Seeds 5000 and above were used only for tuning and are never reported.

## Files
| File | Purpose |
| --- | --- |
| `crwban.py` | The simulator: Markov PU channels, traffic classes, reward (Eq. 3), policies (proposed, partitioned UCB, UCB1 variants, LinUCB, Thompson sampling, Q-learning, rho-rand), gateway, metrics. |
| `final_lib.py` | Helpers that collect metrics and run paired t-tests. |
| `final_A1.py`, `final_A2.py` | Single-SU experiments (Sections 5.1-5.7). |
| `final_B.py` | Multi-SU experiments (Section 6). |
| `final_C.py` | Sensitivity study (Section 5.8). |
| `make_data_figures.py`, `make_diagrams.py`, `make_graphical_abstract.py` | Figures. |
| `facts.py` | Prints the numbers quoted in the text. |
| `tune_baselines.py`, `tune2.py` | Hyperparameter grids of LinUCB and Q-learning (calibration seeds). |
| `weights_sens.py`, `check_prop1.py`, `step4_checks.py` | Weight sensitivity, numerical check of Proposition 1, gateway optimality gap. |
| `early_calibration/` | The first-generation simulator and the scripts that chose the persistence, kappa and kappa_E on calibration seeds. They use their own module `crwban_sim.py`. |
| `results/` | The result files of the reported runs (`final_*.pkl`) and the tuned baseline settings. |

## Default parameters (Table 4 of the paper)
N = 5 channels; busy probabilities 0.11, 0.28, 0.42, 0.64, 0.78; p01 + p10 = 0.5; bandwidths 0.6, 1.0, 0.8, 0.7, 0.9;
emergency probability 0.15; weights 2.5 and 16; kappa_R = 1, kappa_E = 0.25; energy 0.2 + 1.0 + 1.6 per failed attempt; T = 6,000.

