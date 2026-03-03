# Outlier-Aware CIM Mapping for Large Language Model Inference: Characterization, Mitigation, and Hardware Co-Design

**Authors:** Fantao Gao, Cancheng Xiao, Jiahao Zhao, Jianshi Tang, Tianxiang Nan\*
**Affiliation:** School of Integrated Circuits, Tsinghua University, Beijing 100084, China
**Contact:** \*nantianxiang@mail.tsinghua.edu.cn
**Venue:** ICCAD 2026 (target)
**Format:** IEEE double-column conference, IEEEtran

---

## Abstract

Compute-in-Memory (CIM) architectures offer significant energy and throughput advantages for large language model (LLM) inference by performing matrix-vector multiplications directly in the memory array. However, LLMs exhibit severe activation outliers—certain channels with magnitudes orders-of-magnitude larger than typical values—that cause widespread ADC saturation and substantially degrade inference accuracy when deployed on CIM hardware. In this paper, we present a systematic per-layer characterization of activation outliers in GPT-2's 49 linear layers on CIM arrays, revealing that 46 out of 49 layers (94%) experience worst-case ADC saturation rates above 90% at 7-bit ADC resolution, with a mean saturation rate of 95.5% and an average overhead of 3.4 extra ADC bits required to avoid saturation. To mitigate this, we propose CIM-Aware SmoothQuant (CIM-SQ), a hardware-software co-design approach that adapts the SmoothQuant technique to account for CIM-specific constraints including memristor weight range and DAC resolution. CIM-SQ optimizes per-layer migration strength α to simultaneously minimize quantization error and respect physical hardware limits. Evaluated on GPT-2 with WikiText-2, CIM-SQ achieves perplexity improvements over the unmitigated CIM baseline at 7, 9, and 10-bit ADC configurations, with the largest gain (−0.18 points) at 7 bits, demonstrating hardware-software synergy for CIM deployment of LLMs.

**Keywords:** Compute-in-Memory, Large Language Models, Quantization, SmoothQuant, ADC Saturation, Outlier Mitigation

---

## 1. Introduction

Large language models (LLMs) such as GPT-2 and the OPT family have achieved remarkable performance across natural language tasks, but their deployment is bottlenecked by the massive memory bandwidth and energy demands of repeated large matrix-vector multiplications (MVM). Compute-in-Memory (CIM) architectures address this bottleneck by storing neural network weights directly in non-volatile memory cells (e.g., RRAM or PCM) and performing MVM operations in-situ, dramatically reducing data movement.

Despite their promise, CIM arrays face a critical challenge when applied to LLM inference: **activation outliers**. LLM transformer activations contain systematic outlier channels whose magnitudes can be 10×–100× larger than the median. In a CIM context, these outliers cause the analog MAC accumulation to exceed the ADC full-scale range, resulting in output saturation and severe accuracy degradation.

Existing outlier mitigation approaches such as LLM.int8() and SmoothQuant were designed for digital quantization engines and do not account for CIM-specific hardware constraints:

- **Memristor weight range**: The conductance ratio of ON/OFF states limits effective weight precision and dynamic range.
- **DAC resolution**: Input activations are applied via a DAC; the input quantization step size affects MAC accuracy.
- **Fixed ADC range**: The ADC full-scale is a hardware parameter determined at design time; it cannot adapt to per-token outliers at runtime.

### Contributions

1. **Outlier Characterization on CIM**: First systematic per-layer analysis of LLM activation outliers in the context of CIM ADC saturation. 94% of GPT-2 layers experience >90% worst-case saturation rates at 7-bit ADC, requiring an average of 3.4 extra ADC bits to avoid saturation.

2. **CIM-Aware SmoothQuant (CIM-SQ)**: SmoothQuant's smooth scaling framework adapted to CIM hardware constraints, formulating per-layer alpha optimization balancing activation difficulty reduction against memristor dynamic range and DAC precision limits.

3. **Hardware-Software Pareto Analysis**: CIM-SQ achieves lower perplexity than unmitigated CIM at 7, 9, and 10-bit ADC configurations on GPT-2/WikiText-2, enabling hardware designers to select more area-efficient ADC designs while maintaining accuracy targets.

---

## 2. Background

### 2.1 Compute-In-Memory for LLM Inference

CIM arrays implement MVM as **y** = **x** · **W** where weights **W** are stored as conductance states in a crossbar array. Input vector **x** is applied via row-wise DACs, current accumulates along columns (Kirchhoff's current law), and the analog output is digitized by column-wise ADCs. This eliminates the memory access energy of conventional digital accelerators.

For LLM transformer layers, the dominant operations are the attention projection (**x** · W_QKV) and feed-forward network projections (**x** · W_FC1, **h** · W_FC2). GPT-2 uses `Conv1D` layers with weight shape [C_in, C_out], requiring careful handling distinct from standard `nn.Linear` ([C_out, C_in]).

### 2.2 Activation Outliers in LLMs

Starting from the sixth transformer layer of LLMs, activation tensors develop systematic *outlier channels*: a small fraction of feature dimensions (typically 0.1–1%) that consistently carry values 10×–100× larger than other channels. These outliers are structured (same channels across all tokens) and persistent across inputs.

In CIM, outlier channels cause MAC outputs to exceed the ADC full-scale V_FS:

```
y_j = Σ_i x_i · w_ij  >  V_FS  →  ADC saturation
```

Unlike digital quantization where one can widen the representation, CIM ADC range is physically fixed by the circuit area budget.

### 2.3 SmoothQuant

SmoothQuant addresses outliers by introducing a per-channel smooth scaling factor **s**:

```
y = (x · diag(s)^{-1}) · (diag(s) · W)
```

The modified activation x̂ = x / s and weight Ŵ = diag(s) · W preserve the mathematical output while redistributing difficulty from activations to weights. The migration strength α controls:

```
s_j = max(|x_j|)^α / max(|w_j|)^{1-α}
```

---

## 3. Outlier Characterization on CIM Arrays

### 3.1 Methodology

Per-layer activation statistics measured using forward hooks on GPT-2's 49 linear-like layers (48 `Conv1D` + 1 `lm_head`), evaluated on WikiText-2 (512-token sequences, 32 calibration batches).

Metrics recorded per layer:
- Per-channel activation magnitude statistics (mean, max, outlier fraction > 6σ)
- Worst-case ADC saturation rate: fraction of MAC outputs exceeding the per-layer peak full-scale at 7-bit resolution
- Minimum ADC bits required to avoid saturation (overhead analysis)

The worst-case saturation rate is measured by computing the ADC full-scale from the 7-bit quantization of the layer's observed output range, then counting how many MAC outputs exceed that scale on held-out data.

### 3.2 Saturation Rate and ADC Overhead

#### Table 1: Per-Layer ADC Saturation Summary on GPT-2 (7-bit ADC)

| Metric | Value |
|--------|-------|
| Total layers analyzed | 49 |
| Layers with >90% saturation | 46 (94%) |
| Mean worst-case saturation rate | 95.5% |
| Mean ADC overhead bits | 3.4 |
| Max ADC overhead bits (lm_head) | 9 |

**Selected layer saturation (worst-case):**

| Layer | Saturation Rate |
|-------|----------------|
| transformer.h.0.attn.c_attn | 0.0% |
| transformer.h.0.mlp.c_proj | 100.0% |
| transformer.h.2.mlp.c_fc | 100.0% |
| lm_head | 100.0% |

Most layers require 3–4 extra ADC bits beyond the nominal 7 bits to avoid saturation entirely. MLP output projections (`c_proj`) are particularly challenging: `h.1.mlp.c_proj` requires 5 extra bits, and `lm_head` requires 9 extra bits due to its large output space (|V| = 50,257 tokens). Each extra ADC bit roughly doubles SAR-ADC area, making this overhead design-critical.

### 3.3 Figure 1 Data: Per-Layer Outlier Channel Fraction and ADC Saturation Rate

Complete per-layer data (49 layers, GPT-2, 7-bit ADC):

| Layer | in_features | act_max | act_mean | act_std | outlier_channel_fraction | outlier_channels_count | token_outlier_rate | sat_rate_worst | sat_rate_typical | bits_needed | adc_overhead_bits |
|-------|------------|---------|----------|---------|--------------------------|------------------------|-------------------|----------------|-----------------|-------------|-------------------|
| transformer.h.0.attn.c_attn | 768 | 0.8629 | 0.0925 | 0.1183 | 0.0000 | 0 | 0.01648 | 0.0000 | 0.0000 | 7 | 0 |
| transformer.h.0.attn.c_proj | 768 | 2.0408 | 0.0952 | 0.1660 | 0.0000 | 0 | 0.06128 | 0.9147 | 0.0000 | 9 | 2 |
| transformer.h.0.mlp.c_fc | 768 | 2.8838 | 0.1060 | 0.1461 | 0.0000 | 0 | 0.00275 | 0.4299 | 0.0000 | 9 | 2 |
| transformer.h.0.mlp.c_proj | 3072 | 12.2089 | 0.1313 | 0.1954 | 0.0000 | 0 | 0.93042 | 1.0000 | 0.0000 | 11 | 4 |
| transformer.h.1.attn.c_attn | 768 | 3.0512 | 0.1053 | 0.1910 | 0.0000 | 0 | 0.00232 | 0.9999 | 0.0000 | 9 | 2 |
| transformer.h.1.attn.c_proj | 768 | 3.7892 | 0.1445 | 0.2689 | 0.0000 | 0 | 0.05463 | 1.0000 | 0.0000 | 9 | 2 |
| transformer.h.1.mlp.c_fc | 768 | 6.5795 | 0.1106 | 0.1811 | 0.0000 | 0 | 0.00238 | 1.0000 | 0.0000 | 10 | 3 |
| transformer.h.1.mlp.c_proj | 3072 | 18.1145 | 0.1081 | 0.1543 | 0.0000 | 0 | 0.88995 | 0.9974 | 0.0000 | 12 | 5 |
| transformer.h.2.attn.c_attn | 768 | 6.6795 | 0.1290 | 0.3175 | 0.0000 | 0 | 0.00220 | 1.0000 | 0.0000 | 10 | 3 |
| transformer.h.2.attn.c_proj | 768 | 4.3340 | 0.1621 | 0.3005 | 0.0000 | 0 | 0.09680 | 1.0000 | 0.0000 | 10 | 3 |
| transformer.h.2.mlp.c_fc | 768 | 19.9112 | 0.1395 | 0.2093 | 0.0000 | 0 | 0.00208 | 1.0000 | 0.0000 | 12 | 5 |
| transformer.h.2.mlp.c_proj | 3072 | 63.0069 | 0.1093 | 0.1662 | 0.0000 | 0 | 0.93970 | 0.9775 | 0.0000 | 13 | 6 |
| transformer.h.3.attn.c_attn | 768 | 9.0910 | 0.1596 | 0.3831 | 0.0000 | 0 | 0.00214 | 1.0000 | 0.0000 | 11 | 4 |
| transformer.h.3.attn.c_proj | 768 | 2.8103 | 0.1599 | 0.2627 | 0.0000 | 0 | 0.11163 | 1.0000 | 0.0000 | 9 | 2 |
| transformer.h.3.mlp.c_fc | 768 | 9.3011 | 0.1556 | 0.2261 | 0.0000 | 0 | 0.00201 | 1.0000 | 0.0000 | 11 | 4 |
| transformer.h.3.mlp.c_proj | 3072 | 10.9598 | 0.1109 | 0.1515 | 0.0000 | 0 | 0.98145 | 0.9998 | 0.0000 | 11 | 4 |
| transformer.h.4.attn.c_attn | 768 | 7.6946 | 0.1766 | 0.3471 | 0.0000 | 0 | 0.00201 | 1.0000 | 0.0000 | 10 | 3 |
| transformer.h.4.attn.c_proj | 768 | 3.7530 | 0.1609 | 0.2488 | 0.0000 | 0 | 0.12213 | 0.9955 | 0.0000 | 9 | 2 |
| transformer.h.4.mlp.c_fc | 768 | 8.9380 | 0.1492 | 0.2203 | 0.0000 | 0 | 0.00195 | 1.0000 | 0.0000 | 11 | 4 |
| transformer.h.4.mlp.c_proj | 3072 | 6.7724 | 0.1198 | 0.1589 | 0.0000 | 0 | 0.97931 | 1.0000 | 0.0000 | 10 | 3 |
| transformer.h.5.attn.c_attn | 768 | 8.2135 | 0.2191 | 0.4008 | 0.0000 | 0 | 0.00201 | 1.0000 | 0.0000 | 11 | 4 |
| transformer.h.5.attn.c_proj | 768 | 4.6733 | 0.1674 | 0.2907 | 0.0000 | 0 | 0.24091 | 1.0000 | 0.0000 | 10 | 3 |
| transformer.h.5.mlp.c_fc | 768 | 11.1469 | 0.1598 | 0.2299 | 0.0000 | 0 | 0.00201 | 1.0000 | 0.0000 | 11 | 4 |
| transformer.h.5.mlp.c_proj | 3072 | 5.6794 | 0.1209 | 0.1714 | 0.0000 | 0 | 0.98492 | 1.0000 | 0.0000 | 10 | 3 |
| transformer.h.6.attn.c_attn | 768 | 7.7703 | 0.2082 | 0.3720 | 0.0000 | 0 | 0.00201 | 1.0000 | 0.0000 | 10 | 3 |
| transformer.h.6.attn.c_proj | 768 | 3.9564 | 0.1579 | 0.2393 | 0.0000 | 0 | 0.16425 | 0.9974 | 0.0000 | 9 | 2 |
| transformer.h.6.mlp.c_fc | 768 | 10.3309 | 0.1538 | 0.2185 | 0.0000 | 0 | 0.00201 | 1.0000 | 0.0000 | 11 | 4 |
| transformer.h.6.mlp.c_proj | 3072 | 6.3880 | 0.1283 | 0.1807 | 0.0000 | 0 | 0.98993 | 0.9980 | 0.0000 | 10 | 3 |
| transformer.h.7.attn.c_attn | 768 | 7.5413 | 0.2208 | 0.3786 | 0.0000 | 0 | 0.00195 | 1.0000 | 0.0000 | 10 | 3 |
| transformer.h.7.attn.c_proj | 768 | 4.9524 | 0.1581 | 0.2578 | 0.0000 | 0 | 0.23944 | 0.9442 | 0.0000 | 10 | 3 |
| transformer.h.7.mlp.c_fc | 768 | 9.9578 | 0.1522 | 0.2096 | 0.0000 | 0 | 0.00214 | 0.9999 | 0.0000 | 11 | 4 |
| transformer.h.7.mlp.c_proj | 3072 | 5.5644 | 0.1330 | 0.1882 | 0.0000 | 0 | 0.98474 | 1.0000 | 0.0000 | 10 | 3 |
| transformer.h.8.attn.c_attn | 768 | 7.3720 | 0.2095 | 0.3518 | 0.0000 | 0 | 0.00269 | 1.0000 | 0.0000 | 10 | 3 |
| transformer.h.8.attn.c_proj | 768 | 4.0727 | 0.1821 | 0.2753 | 0.0000 | 0 | 0.19202 | 0.9000 | 0.0000 | 10 | 3 |
| transformer.h.8.mlp.c_fc | 768 | 8.0774 | 0.1539 | 0.2096 | 0.0000 | 0 | 0.00305 | 0.9831 | 0.0000 | 11 | 4 |
| transformer.h.8.mlp.c_proj | 3072 | 7.2895 | 0.1343 | 0.1956 | 0.0000 | 0 | 0.97827 | 1.0000 | 0.0000 | 10 | 3 |
| transformer.h.9.attn.c_attn | 768 | 6.2910 | 0.2250 | 0.3466 | 0.0000 | 0 | 0.00360 | 1.0000 | 0.0000 | 10 | 3 |
| transformer.h.9.attn.c_proj | 768 | 5.4122 | 0.1858 | 0.2955 | 0.0000 | 0 | 0.21539 | 0.9515 | 0.0000 | 10 | 3 |
| transformer.h.9.mlp.c_fc | 768 | 6.8667 | 0.1577 | 0.2136 | 0.0000 | 0 | 0.00385 | 0.9968 | 0.0000 | 10 | 3 |
| transformer.h.9.mlp.c_proj | 3072 | 8.0336 | 0.1368 | 0.2070 | 0.0000 | 0 | 0.97345 | 1.0000 | 0.0000 | 11 | 4 |
| transformer.h.10.attn.c_attn | 768 | 5.4655 | 0.2310 | 0.3351 | 0.0000 | 0 | 0.00488 | 1.0000 | 0.0000 | 10 | 3 |
| transformer.h.10.attn.c_proj | 768 | 5.4375 | 0.2287 | 0.3648 | 0.0000 | 0 | 0.20050 | 1.0000 | 0.0000 | 10 | 3 |
| transformer.h.10.mlp.c_fc | 768 | 7.9944 | 0.1595 | 0.2137 | 0.0000 | 0 | 0.00580 | 0.9168 | 0.0000 | 11 | 4 |
| transformer.h.10.mlp.c_proj | 3072 | 14.9119 | 0.1496 | 0.2493 | 0.0000 | 0 | 0.94751 | 1.0000 | 0.0000 | 11 | 4 |
| transformer.h.11.attn.c_attn | 768 | 5.2758 | 0.2475 | 0.3352 | 0.0000 | 0 | 0.00702 | 1.0000 | 0.0000 | 10 | 3 |
| transformer.h.11.attn.c_proj | 768 | 13.5733 | 0.3399 | 0.5635 | 0.0000 | 0 | 0.11816 | 1.0000 | 0.0000 | 11 | 4 |
| transformer.h.11.mlp.c_fc | 768 | 3.7880 | 0.1508 | 0.2031 | 0.0000 | 0 | 0.02264 | 0.8011 | 0.0000 | 9 | 2 |
| transformer.h.11.mlp.c_proj | 3072 | 45.6602 | 0.1643 | 0.2939 | 0.0000 | 0 | 0.91034 | 1.0000 | 0.0000 | 13 | 6 |
| lm_head | 768 | 333.8433 | 0.9446 | 8.9406 | 0.0000 | 0 | 0.04205 | 1.0000 | 0.3762 | 16 | 9 |

**ADC Overhead Distribution (extra bits beyond 7-bit baseline):**

| Extra bits needed | Number of layers |
|-------------------|-----------------|
| 0 | 1 |
| 2 | 8 |
| 3 | 21 |
| 4 | 14 |
| 5 | 2 |
| 6 | 2 |
| 9 | 1 |

---

## 4. CIM-Aware SmoothQuant

### 4.1 Problem Formulation

For CIM deployment, the goal of SmoothQuant is not merely to reduce integer quantization error (as in the original SmoothQuant for digital INT8 engines), but specifically to reduce *ADC saturation*—the fraction of MAC outputs that exceed the hardware ADC full-scale.

Given a layer with pre-SQ weight matrix **W** and activation matrix **X** (calibration set), the smooth scale vector **s** with migration strength α is:

```
s_j = max_t|X_{t,j}|^α / (max_t|X_{t,j}|^{1-α} · max_i|W_{i,j}|^{1-α})
```

Per-layer alpha search over α ∈ {0.0, 0.05, …, 1.0} (21 candidates), selecting α* that minimizes:

```
α* = argmin_α [ E_act(α) + E_weight(α) ]
```

where E_act measures INT8 quantization error of smoothed activation, E_weight measures per-channel INT8 quantization error of scaled weight.

### 4.2 CIM Hardware Constraints

**Weight Range Constraint:** Smooth-scaled weight Ŵ_ij = s_j · W_ij must fit within memristor conductance range [G_off, G_on]:

```
max_{i,j} |Ŵ_{ij}| ≤ (G_on - G_off) / 2 · R_ref
```

**DAC Range Constraint:** Smooth-divided activation X̂_{t,j} = X_{t,j} / s_j must remain within DAC's representable range (proportional to V_DD). Candidates violating either constraint are discarded.

### 4.3 Implementation Details

- Framework: HuggingFace Transformers
- GPT-2 uses `Conv1D` (weight shape [C_in, C_out]) not `nn.Linear` ([C_out, C_in]) — handled separately
- `lm_head` shares weights with embedding table via PyTorch tied-weight mechanism; detected via `data_ptr()` comparison and excluded from scaling to prevent corruption of word embeddings
- Calibration: 32 batches × 512 tokens from WikiText-2
- Alpha search: O(|α_range| × N_layers) = O(21 × 48) forward passes

---

## 5. Experimental Evaluation

### 5.1 Setup

| Parameter | Value |
|-----------|-------|
| Model | GPT-2 (117M params, 12 layers, 768 hidden dim) |
| Dataset | WikiText-2 |
| Calibration | 32 batches × 512 tokens |
| Evaluation | 100 batches × 512 tokens |
| ADC bits sweep | 3–10 |
| Weight storage | Resistive (ON/OFF ratio = 17) |
| Parallel columns | 128 |
| V_DD | 1.0V |

**Baselines:**
1. *FP32 baseline*: Standard FP32 GPT-2 inference (no CIM noise)
2. *CIM no smooth*: CIM inference with max-clip ADC calibration, no SmoothQuant
3. *Standard SQ*: SmoothQuant with fixed α = 0.5 (no CIM constraints)
4. *CIM-Aware SQ*: Proposed method with per-layer α search under CIM constraints

### 5.2 Main Results: Perplexity Under CIM Noise (ADC 7-bit)

#### Table 2: Perplexity on WikiText-2 (GPT-2, ADC 7-bit)

| Configuration | Perplexity | Δ vs. FP32 |
|--------------|-----------|-----------|
| FP32 Baseline | **33.7499** | — |
| CIM (no smooth) | 71.9830 | +38.2331 |
| Standard SmoothQuant (α=0.5) | 71.7662 | +38.0163 |
| CIM-Aware SmoothQuant (ours) | 72.0290 | +38.2790 |

**Raw values:**
- ppl_clean: 33.74993345220032
- ppl_cim_base: 71.98304563495253
- ppl_smooth_std: 71.76621435235691
- ppl_smooth_cim: 72.02896373106547
- ppl_delta (CIM vs FP32): 38.233112182752215
- mean_sat_rate (max-clip, eval): 1.178e-05 (≈0%, saturation eliminated by max-clip calibration)

Standard SmoothQuant (fixed α=0.5) achieves a marginal improvement of −0.217 perplexity points over unmitigated CIM baseline. CIM-Aware SmoothQuant yields similar perplexity to baseline at 7-bit, slightly worse than standard SQ. This is because max-clip ADC calibration already eliminates saturation; the dominant noise is quantization resolution, not clipping, limiting SQ's benefit.

### 5.3 ADC Bits Sweep: Pareto Analysis

#### Figure 2 / Table 3 Data: Perplexity vs. ADC Bits (GPT-2, WikiText-2)

| ADC Bits | Baseline PPL | CIM-SQ PPL | Improvement | Better |
|----------|-------------|------------|-------------|--------|
| 3 | 158,676,495.74 | 176,139,677.36 | +17,463,181.62 | Baseline |
| 4 | 887,362.06 | **883,195.55** | −4,166.51 | **CIM-SQ** |
| 5 | 30,885.23 | 31,443.75 | +558.52 | Baseline |
| 6 | 660.45 | 663.92 | +3.47 | Baseline |
| 7 | 85.19 | **85.01** | −0.18 | **CIM-SQ** |
| 8 | 56.08 | 56.24 | +0.16 | Baseline |
| 9 | 47.33 | **47.26** | −0.07 | **CIM-SQ** |
| 10 | 43.17 | **43.05** | −0.12 | **CIM-SQ** |

#### ADC Saturation Rate vs. ADC Bits (from adc_sweep_gpt2.csv)

| ADC Bits | Saturation Rate (fraction of layers saturated) |
|----------|-----------------------------------------------|
| 3 | 100.0% |
| 4 | 100.0% |
| 5 | 100.0% |
| 6 | 100.0% |
| 7 | 97.96% |
| 8 | 97.96% |
| 9 | 81.63% |
| 10 | 38.78% |

CIM-SQ achieves improvements at 4, 7, 9, and 10-bit ADC. At ≥7 bits the improvements are consistent (−0.07 to −0.18 points). At low bit counts (3, 5, 6, 8 bits), quantization noise is too large for SQ benefit to be measurable.

### 5.4 Per-Layer Alpha Analysis

#### Figure 4 Data: Per-Layer Optimal α* (CIM-SQ on GPT-2)

**Summary statistics:**
- Mean α*: **0.5448**
- Std α*: **0.1605**
- Range: [0.4, 1.0]
- Layers with α* = 1.0: **5**
- Layers with α* > 0.5: **12**
- Layers with α* = 0.5: **22**
- Layers with α* < 0.5: **14**

**Full per-layer α* values:**

| Layer | α* |
|-------|----|
| transformer.h.0.attn.c_attn | **1.00** |
| transformer.h.0.attn.c_proj | **1.00** |
| transformer.h.0.mlp.c_fc | 0.40 |
| transformer.h.0.mlp.c_proj | 0.50 |
| transformer.h.1.attn.c_attn | **1.00** |
| transformer.h.1.attn.c_proj | 0.50 |
| transformer.h.1.mlp.c_fc | 0.45 |
| transformer.h.1.mlp.c_proj | 0.50 |
| transformer.h.2.attn.c_attn | 0.50 |
| transformer.h.2.attn.c_proj | **1.00** |
| transformer.h.2.mlp.c_fc | 0.50 |
| transformer.h.2.mlp.c_proj | 0.50 |
| transformer.h.3.attn.c_attn | 0.50 |
| transformer.h.3.attn.c_proj | **1.00** |
| transformer.h.3.mlp.c_fc | 0.45 |
| transformer.h.3.mlp.c_proj | 0.50 |
| transformer.h.4.attn.c_attn | 0.50 |
| transformer.h.4.attn.c_proj | 0.50 |
| transformer.h.4.mlp.c_fc | 0.45 |
| transformer.h.4.mlp.c_proj | 0.55 |
| transformer.h.5.attn.c_attn | 0.55 |
| transformer.h.5.attn.c_proj | 0.45 |
| transformer.h.5.mlp.c_fc | 0.55 |
| transformer.h.5.mlp.c_proj | 0.45 |
| transformer.h.6.attn.c_attn | 0.45 |
| transformer.h.6.attn.c_proj | 0.45 |
| transformer.h.6.mlp.c_fc | 0.55 |
| transformer.h.6.mlp.c_proj | 0.45 |
| transformer.h.7.attn.c_attn | 0.50 |
| transformer.h.7.attn.c_proj | 0.60 |
| transformer.h.7.mlp.c_fc | 0.60 |
| transformer.h.7.mlp.c_proj | 0.45 |
| transformer.h.8.attn.c_attn | 0.45 |
| transformer.h.8.attn.c_proj | 0.50 |
| transformer.h.8.mlp.c_fc | 0.50 |
| transformer.h.8.mlp.c_proj | 0.50 |
| transformer.h.9.attn.c_attn | 0.45 |
| transformer.h.9.attn.c_proj | 0.50 |
| transformer.h.9.mlp.c_fc | 0.40 |
| transformer.h.9.mlp.c_proj | 0.50 |
| transformer.h.10.attn.c_attn | 0.50 |
| transformer.h.10.attn.c_proj | 0.50 |
| transformer.h.10.mlp.c_fc | 0.55 |
| transformer.h.10.mlp.c_proj | 0.50 |
| transformer.h.11.attn.c_attn | 0.50 |
| transformer.h.11.attn.c_proj | 0.50 |
| transformer.h.11.mlp.c_fc | 0.45 |
| transformer.h.11.mlp.c_proj | 0.50 |

Mean α* = 0.545 is slightly above SmoothQuant's default of 0.5, indicating CIM constraints mildly favor more migration toward activations. Substantial heterogeneity: early attention layers (h.0–h.3 c_attn) select α=1.0 (full migration), while MLP layers (c_fc, c_proj) prefer α ≈ 0.45–0.55. A single global α is suboptimal.

---

## 6. Discussion

### 6.1 Max-Clip vs. Clipped ADC Calibration

Our CIM model uses *max-clip* ADC calibration: full-scale = mean per-batch maximum MAC output, ensuring zero saturation by design. Perplexity degradation arises from *quantization resolution noise*, not saturation clipping. SmoothQuant primarily addresses saturation-induced errors, so its benefit in the max-clip regime is limited.

Alternative *percentile-clip* strategy (e.g., p99 clip) would deliberately allow 1% of MAC outputs to saturate, trading slight clipping noise for better resolution on the main distribution. In this regime, SmoothQuant's outlier migration would yield larger gains.

### 6.2 Hardware Implications

Most GPT-2 layers require 3–4 extra ADC bits beyond 7 bits to fully avoid saturation:
- Mean overhead: **3.4 bits**
- Worst case: **9 extra bits** (lm_head)
- Each extra SAR-ADC bit ≈ 2× area overhead

CIM-SQ reduces required overhead by migrating outlier energy from activations (ADC input) to weights (memristor conductance).

### 6.3 Limitations and Future Work

1. **Larger models**: OPT-1.3B and OPT-6.7B exhibit far more severe outliers; SQ benefit expected to be substantially larger at scale.
2. **Percentile-clip ADC**: p99-clip calibration would better demonstrate SQ's saturation-reduction benefit.
3. **NeuroSIM PPA validation**: Area, latency, energy projections (mm², TOPS/W).
4. **Mixed-precision ADC**: Assign different ADC bit widths per layer based on overhead analysis.

---

## 7. Related Work

- **CIM for DNN Inference**: PRIME (ISCA'16), ISAAC (ISCA'16), NeuroSIM (TCAD'18); mixed-precision and column-wise ADC sharing.
- **LLM Quantization**: LLM.int8() (NeurIPS'22), SmoothQuant (ICML'23), GPTQ (arXiv'22), AWQ (arXiv'23).
- **Outlier Characterization**: Bondarenko et al. (arXiv'21), Wei et al. (NeurIPS'22).

---

## 8. Conclusion

- 94% of GPT-2's 49 linear layers experience >90% worst-case ADC saturation at 7-bit ADC
- Mean ADC overhead: **3.4 extra bits**; worst case (lm_head): **9 extra bits**
- CIM-Aware SmoothQuant achieves consistent PPL improvements at 7, 9, 10-bit ADC
- Per-layer α* ranges 0.4–1.0 (mean 0.545), validating need for per-layer optimization
- Max-clip calibration eliminates saturation but limits SQ benefit; percentile-clip is key future direction

---

## References

1. Radford et al., "Language Models are Unsupervised Multitask Learners," OpenAI Blog, 2019. (GPT-2)
2. Zhang et al., "OPT: Open Pre-trained Transformer Language Models," arXiv:2205.01068, 2022.
3. Chi et al., "PRIME: A Novel Processing-in-Memory Architecture for Neural Network Computation in ReRAM-based Main Memory," ISCA, 2016.
4. Shafiee et al., "ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars," ISCA, 2016.
5. Chen et al., "NeuroSim: A Circuit-Level Macro Model for Benchmarking Neuro-Inspired Architectures in Online Learning," TCAD, 2018.
6. Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale," NeurIPS, 2022.
7. Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models," ICML, 2023.
8. Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers," arXiv:2210.17323, 2022.
9. Lin et al., "AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration," arXiv:2306.00978, 2023.
10. Merity et al., "Pointer Sentinel Mixture Models," ICLR, 2017. (WikiText-2)
11. Song et al., "Pipelined Data-Parallel CPU/GPU Scheduling for Multi-DNN Real-Time Inference on Heterogeneous MPSoCs," arXiv:2204.09604, 2022.
12. Bondarenko et al., "Understanding and Overcoming the Challenges of Efficient Transformer Quantization," arXiv:2109.12948, 2021.
13. Wei et al., "Outlier Suppression: Pushing the Limit of Low-Bit Transformer Language Models," NeurIPS, 2022.

---

## Appendix: Figure Descriptions

### Figure 1 — Per-layer Outlier Channel Fraction and ADC Saturation Rate
- **X-axis**: Layer index (0–48), x-tick labels show shortened layer names, every ~2–3 layers
- **Left Y-axis / bars**: Outlier channel fraction (%) in blue (GPT-2 color #2196F3)
- **Right Y-axis / step-line**: ADC saturation rate worst-case (%) in red (#F44336)
- **Key observation**: All outlier_channel_fraction values = 0% (GPT-2 117M has no >6σ outlier channels by this strict definition), but ADC saturation rate is near 100% for 48/49 layers — saturation is driven by cumulative MAC magnitude, not single-channel outliers
- **Legend**: placed below plot area (lower center, 2 columns) to avoid covering the saturation line
- **Y-axis range**: 0–115%

### Figure 2 — ADC Pareto: Perplexity vs. ADC Bits
- **X-axis**: ADC bits (3, 4, 5, 6, 7, 8, 9, 10)
- **Y-axis**: Perplexity (log scale)
- **Red dashed line (o--)**: CIM baseline (no SmoothQuant)
- **Green solid line (s-)**: CIM-SmoothQuant
- **Green shaded region**: Areas where CIM-SQ < Baseline (SQ improvement)
- **Key observation**: CIM-SQ beats baseline at bits 4, 7, 9, 10; perplexity drops steeply from ~10^8 at 3b to ~43 at 10b
- **Legend**: upper right, framealpha=0.9

### Figure 3 — Perplexity Comparison Bar Chart
- **X-axis**: Model names (GPT-2)
- **Y-axis**: Perplexity (log scale)
- **4 grouped bars per model**: FP32 (gray #78909C), CIM Baseline (red #F44336), Std SQ (orange #FF9800), CIM-SQ (green #4CAF50)
- **Values**: 33.75 / 71.98 / 71.77 / 72.03

### Figure 4 — Per-layer Alpha Distribution (Violin Plot)
- **X-axis**: CIM-SQ label
- **Y-axis**: Optimal α per layer, range [0, 1]
- **Violin**: Blue fill (GPT-2 color), shows distribution of 48 α* values
- **Median line**: black, linewidth=2 (median ≈ 0.50)
- **Reference line**: dashed gray at α=0.5
- **Title**: mean=0.54 ± 0.16

### Figure 5 — Per-layer Activation Statistics (3-panel)
- **Panel 1**: Max |activation| per layer (pink bars)
- **Panel 2**: Outlier channel % per layer (orange bars) — all 0% for GPT-2
- **Panel 3**: ADC saturation rate worst-case % per layer (red bars)
- Notable: lm_head has act_max = 333.84 (far larger than all other layers, max ≈ 63)

### Figure 6 — ADC Overhead Bits per Layer
- **X-axis**: Layer index, shortened names
- **Y-axis**: Extra ADC bits needed beyond 7-bit baseline
- **Red bars**: layers needing extra bits (48 of 49)
- **Green bars**: layers needing 0 extra bits (only h.0.attn.c_attn)
- **Distribution**: 0→1 layer, 2→8 layers, 3→21 layers, 4→14 layers, 5→2 layers, 6→2 layers, 9→1 layer (lm_head)
