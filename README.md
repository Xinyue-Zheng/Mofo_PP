# MoFo: Empowering Long-term Time Series Forecasting with Periodic Pattern Modeling (NeurIPS 2025)
This is the official repository of our NeurIPS 2025 Paper. This paper introduces MoFo, which interprets periodicity as both the correlation of period-aligned time steps and the trend of period-offset time steps. We first design period-structured patches—2D tensors generated through discrete sampling—where each row contains only period-aligned time steps, enabling direct modeling of periodic correlations. Period-offset time steps within a cycle are aligned in columns. To capture trends across these offset time steps, we introduce a period-aware modulator. This modulator introduces an adaptive strong inductive bias through a regulated relaxation function, encouraging the model to generate attention coefficients that align with periodic trends. This function is end-to-end trainable, enabling the model to adaptively capture the distinct periodic patterns across diverse datasets. Extensive empirical results on popular benchmark datasets demonstrate that MoFo achieves competitive performance compared to 17 advanced baselines, while offering up to 14x memory efficiency gain and 10x faster training speed.

<img src='MoFo.png' alt='Motivation of MoFo'>

## 1. Introduction about the code
### 1.1 Coding Framework
All of our experiments are running on the [TFB](https://github.com/decisionintelligence/TFB) coding framework. To run MoFo, you need to configure your environment and datasets according to their requirements.

`Since the framework, details, and environment of TFB have been updated, the current MoFo code is no longer applicable. With the help of the official TFB team, we will promptly update MoFo's code and results to align with the latest TFB framework before the final revision deadline for NeurIPS 2025.`

<br>

## 2. Environmental Requirments
The experiment requires the same environment as [TFB](https://github.com/decisionintelligence/TFB).

<br>

## 3. Reproduction of the Long-term Time Series Forecasting
The experimental running on MoFo are integrated within file `/scripts/MoFo_Example.sh`. You can run it through the following commands,
```
sh MoFo_Example.sh
```
Other hyperparameters can be seen in the Appendix of Paper.

## 4. MoFo++ Ablation Scripts

We provide six ablation scripts that progressively add the two MoFo++ extensions (adaptive per-channel period estimation and channel attention). All scripts sweep over prediction horizons {96, 192, 336, 720}.

| Script | Dataset | Adaptive Period | Channel Attention | Description |
|---|---|---|---|---|
| `run_fixed.sh` | ETTh1 | ✗ | ✗ | Ablation 1 — original MoFo with a fixed global period (`PERIODIC=24`) |
| `run_fixed_weather.sh` | Weather | ✗ | ✗ | Ablation 1 — original MoFo on Weather (`PERIODIC=144`, 10-min intervals) |
| `run_adaptive.sh` | ETTh1 | ✓ | ✗ | Ablation 2 — adaptive per-channel period, no channel attention |
| `run_adaptive_weather.sh` | Weather | ✓ | ✗ | Ablation 2 — adaptive per-channel period on Weather |
| `run_adaptive_cattn.sh` | ETTh1 | ✓ | ✓ | Ablation 3 — full MoFo++ (adaptive period + channel attention, `n_heads_channel=4`) |
| `run_adaptive_cattn_weather.sh` | Weather | ✓ | ✓ | Ablation 3 — full MoFo++ on Weather |

Run any script directly, e.g.:
```bash
sh run_adaptive_cattn.sh      # full MoFo++ on ETTh1
sh run_adaptive_cattn_weather.sh  # full MoFo++ on Weather
```

## 5. Citation
```bibtex
@inproceedings{ma2025mofo, 
  title     =  {MoFo: Empowering Long-term Time Series Forecasting with Periodic Pattern Modeling},
  author    = {Ma, Jiaming and Wang, Binwu and Huang, Qihe and Wang, Guanjun and Wang, Pengkun and Zhou, Zhengyang and Wang, Yang},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```
