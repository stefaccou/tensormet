# Upstream repositories

This folder is empty in the repository. The evaluation reads task data and reference files from three
upstream repositories. Clone them here, under exactly these folder names:

```bash
cd 5_evaluation/COLING/third_party
git clone https://github.com/Sandipan99/POLAR POLAR
git clone https://github.com/harsh19/SPINE spine
git clone https://github.com/SINr-Embeddings/sinr sinr
```

Nothing in them is edited.

| Folder | What is read | Used by |
|---|---|---|
| `POLAR/Downstream Task/*/data/` | the datasets of the downstream tasks | `polar_sweep.py` |
| `POLAR/Antonym_sets/` | the antonym pairs of the POLAR transform | `method_baselines.py` |
| `spine/code/evaluation/intrinsic/word_sim.tab` | WordSim-353 | `polar_sweep.py` |
| `sinr/notebooks/sinrvec_bnc.pk` | the SINr model (BNC) | `method_baselines.py` |

The scripts that `polar_sweep.py` ports are in the same repositories: `POLAR/Downstream Task/*/classify*.py`,
`POLAR/main.ipynb` and `spine/code/evaluation/intrinsic/evaluate_wordSim.py`.
