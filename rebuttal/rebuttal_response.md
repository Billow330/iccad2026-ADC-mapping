# ICCAD 2026 Submission #707 — Author Response

We thank Reviewers #707A–#707D. We are encouraged that the work is seen as
addressing a "critical hardware bottleneck" and as "bridging high-level deep
learning properties ... with circuit-level physics" (#707A), backed by
"extensive analysis and optimization" (#707B), with a "simple and practical
flow" and a "clean comparison" against proxy baselines (#707C), and as an
"interesting empirical finding" (#707D).

A supplementary PDF (Tables S1–S3) accompanies this response. Per the rebuttal
rules the paper is unchanged; the clarifications below are for the camera-ready.
We give Common Responses (CR-1–CR-4) to shared concerns, then per-reviewer
answers.

---

## CR-1. Robustness of the sensitivity finding and status of the mechanism
*(#707A-W1, #707C-W2/Q2, #707D-W3)*

Our central, allocation-relevant claim is **group-level**: FFN layers are more
ADC-sensitive than attention layers, and conventional proxies mispredict this.
The paper's core empirical finding — the **saturation–sensitivity inversion**
(saturation rate is anti-correlated with measured ADC sensitivity, ρ=−0.70) — is
a measured correlation that holds regardless of whether the propagation mechanism
(Fig. 5) is proven; the robustness checks below are supplementary support.
New measurements (Table S2) strengthen it. Sweeping probe depth (7→6/5/4) × 3
calibration seeds, FFN>attention holds in 7/9 single-seed settings; crucially,
**seed-averaged it holds clearly at the small 7→6 point** (FFN +0.044 vs
attention +0.023) and overwhelmingly at 7→4 (3/3, ≈**11×**), with 7→5 the noisy
middle. The
single-seed 7→6 noise is a finite-batch artifact: with a larger profiling budget
(calib 16, eval 60) the 7→6 ordering is stable across seeds. We also measured
propagation diagnostics (output KL divergence, hidden-state drift) and found that
propagation **magnitude** scales with a layer's output width and does not by
itself predict accuracy sensitivity — which is precisely why we measure ΔPPL
**directly** rather than relying on a proxy or on the propagation picture.
Fig. 5 therefore remains an interpretive hypothesis; the allocation flow does
**not** depend on it (it consumes only measured ΔPPL, validated by the
ILP-vs-brute-force optimality check, regret = 0).

## CR-2. Does the ordering survive a stronger weight-quantization floor?
*(#707C-C7, #707D-W1/Q1)*

Yes. ADC-bit allocation is **orthogonal** to weight quantization. Re-profiling on
an INT8 **per-channel**-quantized model (a stronger floor than per-tensor)
preserves the ordering: the FFN aggregate remains more ADC-sensitive than the
attention aggregate and $W_{fc2}$ stays the most-sensitive group (Table S1).
The signal is a property of **where ADC noise enters the transformer**, not an
artifact of a degraded baseline. Better PTQ (GPTQ/AWQ/SmoothQuant) lowers the
absolute PPL floor but does not change the relative ADC sensitivity the ILP
optimizes; the paper already reports SQ+6b as an orthogonal reference, and
combining the two is compatible.

## CR-3. Cross-architecture coverage and scalability
*(#707A-W2, #707C-W3/C8)*

Full profiling→ILP→evaluation is reported on OPT-125M and OPT-1.3B; ordering is
additionally validated on Pythia-410M and Qwen2-7B. #707C-C8 conflates two distinct
asks — *scale* and *modern architecture*: the complete PPL/area/energy/latency/downstream
pipeline is reported on OPT-125M (Sec. 5.3), and full allocation at scale on OPT-1.3B
(Table 4). New: at a usable max-clip operating point, the full profiling→ILP→PPA
pipeline on **Qwen2-7B (Table S3)** yields **34% ADC area+energy saving at +1.8% PPL**
(52% at +4.9%) with **PIQA/BoolQ preserved within ~1.6 pp** of the CIM-8b baseline,
while proxy-blind reduction of the most-sensitive group destroys the model —
confirming the method transfers to a modern non-OPT architecture. We report the **ordering**, which is the
transferable, allocation-relevant signal. The practical payoff scales from tens of
mm² (OPT-125M) to hundreds (OPT-1.3B).

## CR-4. ADC implementation realism: topology/node, overhead, area scaling
*(#707B-W1/C1/Q1, #707D-W2/Q2/Q3)*

- **Topology/node (already in Sec. 5):** 45 nm CMOS, 128×128 RRAM crossbar,
  **MLSA-type ADC**, 1-bit (bit-serial) DAC, INT8 weights with bit-slicing. ADC
  area follows the MLSA model $A_{ADC}\propto M\cdot2^{b}$ (Eq. 1); all areas are
  **NeuroSIM-model estimates**, not silicon.
- **Framing (we adopt #707B's suggestion):** the camera-ready will lead with the
  hardware-agnostic lever — **per-layer ADC bit reduction** — and present area as
  the NeuroSIM consequence, consistent with "silicon validation is future work."
- **Mixed-precision overhead / timing (#707D-W2/Q2):** allocation is **group-level**,
  so a whole functional group shares one bit-width; each array stays internally
  uniform (no per-column heterogeneity), MLSA levels are a fixed $2^{b}$ per
  array, and column reads remain synchronous — so **no per-column multiplexing,
  irregular routing, timing skew, or pipeline stall is introduced**. The only
  extra cost is a small per-array bit-width configuration register; since ADC
  peripherals already dominate the macro, its area, power, and latency are
  negligible against the 20% ADC-area saving and are included in the NeuroSIM
  estimate.
- **Non-exponential scaling (#707D-Q3):** the $2^{b}$ law is specific to
  flash/MLSA ADCs; for an (approximately linear-area) SAR ADC the absolute saving
  shrinks, but the **allocation method and the sensitivity ordering are
  unchanged** — fewer bits on tolerant layers still saves area monotonically.

---

## Reviewer #707A
> *W1: inversion is an intuitive qualitative hypothesis, not a proof.*
See **CR-1**: Fig. 5 stays a hypothesis; the method relies only on measured
sensitivity, and FFN>attention is now shown robust (Table S2).

> *W2: end-to-end allocation focuses on smaller workloads.*
See **CR-3** (OPT-1.3B full allocation + new Qwen2-7B full pipeline, Table S3).

> *W3: when to move from group-level to per-layer control?*
Quantified in **Sec. 3.2**: group-ILP beats per-layer ILP at moderate budgets
(≤30% savings) because group averaging suppresses limited-batch noise; per-layer
wins only at aggressive budgets (>40%). We will state this ~30–40% crossover as
an explicit design rule.

## Reviewer #707B
> *W1: light on ADC physical-implementation details; the precise "20%" should be taken with a grain of salt.*
See **CR-4**: we specify 45 nm CMOS, MLSA-type ADC, 128×128 RRAM, 1-bit DAC, and
bit-sliced INT8, with $A_{ADC}\propto M\cdot2^{b}$ (Eq. 1). All areas are
**NeuroSIM-model estimates, not silicon**; the **relative** 20% saving is set by the
$2^{b}$ law and the per-group column counts, so it is robust to fine layout details
even though an absolute silicon number remains future work.

> *C1 (comment): formulate as a "bit-reduction" benefit rather than an "area" benefit.*
Adopted — see **CR-4**: the camera-ready leads with the hardware-agnostic
**per-layer ADC bit-reduction** and presents area as the NeuroSIM consequence.

> *Q1: provide ADC-circuit/CIM assumptions, or stick to bit-number savings?*
We do **both** — see **CR-4**: the assumptions above are stated explicitly, and we
foreground the bit-reduction result; the relative ADC-area saving follows from the
$2^{b}$ law independent of fine layout details.

## Reviewer #707C
> *W1: PPL is coarse; it does not isolate the source (ADC clipping / activation / residual / logit drift / INT8 interaction).*
PPL is used **deliberately** as the deployment metric, but we isolate the relevant
part: the **INT8-only control** (Sec. 5.1) removes the dominant confound — the
FP32→CIM gap is INT8 weight quantization and ADC reduction adds only ~0.2 PPL — so
our signal is the **ADC-induced ΔPPL on a fixed weight-quant floor**. For the finer
intra-ADC sources, the diagnostics we measured (output-KL, hidden-state drift; CR-1)
do **not** predict sensitivity, so a decomposed proxy is unreliable and we optimize
the directly-measured ΔPPL.

> *W2: evidence mostly group-level PPL; mechanism qualitative; ranking model-dependent (Qwen2 $W_{fc1}$).*
The mechanism (Fig. 5) is an interpretive hypothesis, **not load-bearing** —
allocation consumes only measured ΔPPL. The headline finding (proxy failure /
inversion) is itself a **measured correlation** (saturation vs sensitivity ρ=−0.70),
independent of mechanism. The model-dependence is **architectural** (Qwen2's gated
MLP makes $W_{fc1}$ the sensitive FFN sub-layer); the architecture-independent claim
**FFN>attention** holds on all four models (Table 4) and across probes/seeds (Table S2).

> *W3: validation limited (full allocation mainly OPT); gain over proxies modest.*
OPT-125M gives the full PPA+downstream pipeline and OPT-1.3B (145 layers) the full
allocation **at scale**; Pythia-410M and Qwen2-7B add architecture-diversity ordering
(Qwen2 full PPA + downstream now in Table S3). The gain is modest because
the contribution is the **signal**: proxies protect the wrong layers, so random costs
+2.6%±1.4% at the same budget, and at scale measured sensitivity beats the saturation
proxy at **both** OPT-1.3B budgets (C5/C6).

> *C1: define "sensitivity" precisely.*
Agreed: "PPL-based sensitivity to ADC-bit reduction under the specified CIM
setup," not general layer importance.

> *C2: justify PPL vs token-level NLL.*
See **Q1** (Table S1): the ranking and ILP allocation are identical under NLL (Spearman ρ=1.0).

> *C3: add mechanism diagnostics (MSE/SQNR, residual-norm, KL, drift).*
See **W1**/**W2**/**Q2**/**CR-1**.

> *C4: separate model-specific from general ranking.*
See **W2**/**Q2**.

> *C5/C6: improvement is modest; clarify the contribution.*
The contribution is the **finding that standard proxies protect the wrong
layers**: saturation is anti-correlated (ρ=−0.70) and Hessian uncorrelated
(ρ=0.20) with measured ADC sensitivity. Random allocation costs +2.6%±1.4% at the
same budget, so the **signal, not the solver**, drives allocation quality.
The 30% Hessian closeness on OPT-125M is coincidental — Hessian is essentially
uncorrelated (ρ=0.20) and at 20.5% it is clearly worse (309.7 vs 308.6); at scale
(OPT-1.3B) measured sensitivity beats the saturation proxy at **both** budgets
(−0.06% vs +0.43%; +0.31% vs +0.89%, paper Table 4).

> *C7: combine with GPTQ/AWQ/SmoothQuant; operating-point realism.*
See **CR-2**: the ADC ordering is orthogonal to the weight-quant floor and combines
with stronger PTQ (SQ+6b is already an orthogonal reference).

> *C8: full pipeline on a modern non-OPT model.*
See **CR-3** and **Table S3**: the full profiling→ILP→PPA pipeline now runs on
Qwen2-7B at a usable operating point (34% ADC area+energy saving at +1.8% PPL,
PIQA/BoolQ preserved within ~1.6 pp of CIM-8b).

> *Q1: recompute sensitivity with token-level NLL / cross-entropy?*
**Table S1**: the per-group ranking is identical (**Spearman ρ = 1.0**) and the
group-level ILP allocation is unchanged. Since PPL = exp(mean NLL) is strictly
monotonic with a shared baseline, this invariance is provable; the additive
linear surrogate is therefore well-posed under the more additive NLL loss.

> *Q2: stronger mechanism evidence; why the top group differs OPT vs Qwen2.*
See **CR-1** (diagnostics; Fig. 5 is a hypothesis). The OPT-vs-Qwen2 difference is
**architectural, not contradictory**: Qwen2's gated MLP (SwiGLU) gives $W_{fc1}$ a
different role; the robust, architecture-independent claim is **FFN > attention**
(all four models, paper Table 4), which we separate from the model-specific
identity of the single most-sensitive FFN sub-layer.

## Reviewer #707D
> *W1: baseline PPL (~36 → ~300) is dominated by weight quantization.*
Acknowledged and quantified (INT8-only control). Our target is the **additional**
ADC cost above this floor; **CR-2** shows the ADC ordering is independent of the
floor, and the floor is reducible with orthogonal PTQ.

> *W2: control/routing/mux/timing overheads of mixed-precision.*
See **CR-4**: group-level uniformity avoids per-column multiplexing and irregular
routing; the sole added element is a per-array config register (negligible area,
power, latency), included in the NeuroSIM estimate.

> *W3: mechanism is qualitative; the ILP uses a simplified linear degradation model.*
Mechanism: see **CR-1**. The linear surrogate is **validated, not assumed**:
exhaustive group-level enumeration shows the ILP matches the brute-force optimum
(**regret = 0**, Sec. 4.2 / Table 2), and at moderate budgets the ILP reduces only
1–2 groups, where the per-group measurements are exact by construction.

> *C1: validate on a stronger, practically-usable quantized baseline.*
See **W1**/**CR-2**; and **Table S3** shows the ordering and the allocation's savings
hold at a usable Qwen2-7B operating point (PPL 16).

> *C2: include mixed-precision HW overhead (control, reconfig, mux, routing) in area/power/latency.*
See **W2**/**CR-4**.

> *C3: more rigorous ADC-noise propagation analysis.*
See **W3**/**CR-1**.

> *Q1: does the ordering persist when weight-quant error is reduced?*
Yes — see **CR-2** (INT8 per-channel floor preserves FFN>attention, Table S1).

> *Q2: timing mismatch / pipeline stalls with heterogeneous ADC bit-widths?*
See **CR-4**: group-level allocation keeps each array internally uniform and
column reads synchronous, so no per-column timing skew arises; the only overhead
is a per-array config register, included in the NeuroSIM area.

> *Q3: do benefits remain under non-exponential ADC area scaling?*
See **CR-4**: the method and ordering are unchanged; only the magnitude of the
saving depends on the ADC's area-vs-bit curve (largest for flash/MLSA, smaller
for SAR).

---

We thank the reviewers again. The supplementary results (Tables S1–S3) and the
clarifications address the central concerns — proxy failure, the FFN>attention
ordering and its robustness, metric invariance, weight-quant independence, and
ADC modeling — within the existing contribution.
