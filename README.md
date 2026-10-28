<img src="docs/banner.png" alt="Attention-head taxonomy" width="100%">

Can Adaptive Resonance Theory sort a transformer's attention heads into types without being told what types exist? This pipeline hooks all 256 heads of OLMo-2-1B, describes each head with 14 attention statistics over 600 prompts, and clusters them with ART. It then repeats the process on the SFT and Instruct checkpoints.

A single ART run depended on input order (pairwise ARI across orderings was 0.07–0.11), so the taxonomy comes from 100-run consensus clustering. It is compared against k-means and agglomerative baselines.

| | Result |
|---|---|
| Agreement with heuristic head labels | ARI 0.725 (consensus Fuzzy ART, 8 types) vs 0.403 best baseline |
| Base → SFT | taxonomy ARI 0.11, mean feature drift 0.78 |
| SFT → Instruct (DPO + RL) | taxonomy ARI 0.88, mean feature drift 0.06 |

The heuristic labels are rough, so the first row measures agreement, not accuracy.

## Run

```bash
pip install -r requirements.txt
python scripts/01_extract_features.py --tag base     # needs a GPU; eager attention
python scripts/01_extract_features.py --tag sft --model-id allenai/OLMo-2-0425-1B-SFT
python scripts/01_extract_features.py --tag instruct --model-id allenai/OLMo-2-0425-1B-Instruct
python scripts/05b_consensus_refined.py --tag base
python scripts/08_cross_checkpoint.py --tags base sft instruct
```

Configs live in `config/`, and outputs are written to `results/`.
