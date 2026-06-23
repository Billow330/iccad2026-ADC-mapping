# ICCAD 2026 Submission #707 — Author Response

We thank Reviewers #707A–#707D. We are encouraged that the work is seen as
addressing a "critical hardware bottleneck" and as "bridging high-level deep
learning properties ... with circuit-level physics" (#707A), backed by
"extensive analysis and optimization" (#707B), with a "simple and practical
flow" and a "clean comparison" against proxy baselines (#707C), and as an
"interesting empirical finding" (#707D).

A supplementary PDF (Tables S1–S4) accompanies this response. Per the rebuttal
rules the paper is unchanged; clarifications are for the camera-ready. We give
Common Responses (CR-1–CR-4), then per-reviewer answers.

---

## CR-1. Mechanism of the saturation–sensitivity inversion: from hypothesis to analysis
*(#707A-W1, #707C-W2/C3/Q2, #707D-W3)*

We have upgraded Fig. 5 from a qualitative picture to an **analytical
decomposition**, verified by the diagnostics the reviewers requested (**Table S4**).
A uniform *b*-bit ADC with calibrated full-scale *V*₍FS₎ splits its error into a
**granular** (resolution) term and an **overload/clipping** term:
**D_ADC(b) = D_gran(b) + D_ovl**, with **D_gran ∝ (1−s)·V²₍FS₎·4⁻ᵇ** and **D_ovl
independent of b**. Reducing one bit therefore multiplies **only** the granular term
by 4 — confirmed across **every** group: 7→6 scales D_gran by **4.06 ≈ 4¹** and D_ovl
by **1.00** (Table S4). The perturbation a group receives is thus its granular budget
ΔD_gran, propagated to the loss (Fig. 5), and **two measured facts show why no local
signal can rank it**: (i) ΔD_gran does **not** track saturation — since
D_gran ∝ (1−s)·V²₍FS₎, the high-saturation W_qkv has the **largest** budget (large
V_FS), not the smallest; (ii) **decoupling** — across groups ΔD_gran spans **≈740×**,
yet hidden-state drift spans only **≈1.6×** (output KL ≈2.2×) and is **non-monotone**
in measured ΔPPL, so even output-level diagnostics don't rank loss-level sensitivity.
Saturation (ρ=−0.70) and Hessian (ρ=0.20) each see only one local statistic and miss
this propagation — **so direct ΔPPL profiling is necessary**. The flow consumes only
measured ΔPPL and is provably optimal (regret = 0, Table 2); FFN>attention is robust
across probes/seeds (Table S2, ≈11× at 7→4).

## CR-2. Baseline quality: the floor is orthogonal; the ordering holds at usable points
*(#707C-C7, #707D-W1/C1/Q1)*

The high baseline PPL is a **CIM quantization floor orthogonal to our contribution**:
per-layer ADC bit *allocation* only reallocates bits across layers under that floor,
so our ADC-area savings are **separable** from baseline quality.

**Does the ordering survive when the model is usable? Yes (#707D-Q1/C1).** At usable
operating points it holds emphatically: on **OPT-125M at a max-clip 8b point
(PPL 93)**, W_fc2 is the most sensitive group and FFN (+21.5/layer) ≫ attention
(−1.4) (Table S2 note); and on **Qwen2-7B (PPL 15.6 vs clean ~10)**, FFN>attention
holds with profiling-ILP saving **34% ADC area within ~1.6 pp** downstream (Table S3).
The ordering is a **structural** property of *where* ADC noise enters the residual
stream (CR-1) — also stable across four ADC operating points, INT8 per-channel
(Table S1), and four architectures.

**Stronger weight-PTQ (#707C-C7)** targets a different error source and **composes**
with ADC allocation: SmoothQuant (SQ+6b) is in Table 3, and GPTQ/AWQ + profiling-ILP
is camera-ready.

## CR-3. Cross-architecture coverage and scalability
*(#707A-W2, #707C-W3/C8)*

Full profiling→ILP→evaluation is on OPT-125M and OPT-1.3B; ordering is also
validated on Pythia-410M and Qwen2-7B. #707C-C8 conflates two distinct
asks — *scale* and *modern architecture*: the full PPL/area/energy/latency/downstream
pipeline is on OPT-125M (Sec. 5.3), and full allocation at scale on OPT-1.3B
(Table 4). New: at a usable max-clip operating point, the full profiling→ILP→PPA
pipeline on **Qwen2-7B (Table S3)** yields **34% ADC area+energy saving at +1.8% PPL**,
while proxy-blind reduction destroys the model — confirming transfer to a modern
non-OPT architecture.

## CR-4. ADC implementation realism: topology/node, overhead, area scaling
*(#707B-W1/C1/Q1, #707D-W2/Q2/Q3)*

- **Topology/node (already in Sec. 5):** 45 nm CMOS, 128×128 RRAM crossbar,
  **MLSA-type ADC**, 1-bit (bit-serial) DAC, INT8 weights with bit-slicing. ADC
  area follows the MLSA model $A_{ADC}\propto M\cdot2^{b}$ (Eq. 1); all areas are
  **NeuroSIM-model estimates**, not silicon.
- **Framing (we adopt #707B's suggestion):** the camera-ready will lead with the
  hardware-agnostic lever — **per-layer ADC bit reduction** — and present area as
  the NeuroSIM consequence.
- **Mixed-precision overhead / timing (#707D-W2/Q2):** allocation is **group-level**,
  so each array stays internally uniform, MLSA levels are a fixed $2^{b}$ per
  array, and column reads remain synchronous — so **no per-column multiplexing,
  irregular routing, timing skew, or pipeline stall is introduced**. The only
  extra cost is a small per-array bit-width register — negligible against the ADC
  peripherals that dominate the macro, and included in the NeuroSIM estimate.
- **Non-exponential scaling (#707D-Q3):** the $2^{b}$ law is specific to
  flash/MLSA ADCs; for an (approximately linear-area) SAR ADC the absolute saving
  shrinks, but the **allocation method and the sensitivity ordering are
  unchanged** — still saving area monotonically.

---

## Reviewer #707A
> *W1: inversion is an intuitive qualitative hypothesis, not a proof.*
See **CR-1** and **Table S4**: the inversion now has an analytical decomposition
(bit-width scales only the granular error, verified at 4.06≈4¹) plus a measured
local-error/loss **decoupling** that explains why saturation and Hessian fail.
FFN>attention is also shown robust (Table S2).

> *W2: end-to-end allocation focuses on smaller workloads.*
See **CR-3** (OPT-1.3B full allocation + new Qwen2-7B full pipeline, Table S3).

> *W3: when to move from group-level to per-layer control?*
Quantified in **Sec. 3.2**: group-ILP beats per-layer ILP at moderate budgets
(≤30% savings) because group averaging suppresses limited-batch noise; per-layer
wins only at aggressive budgets (>40%). We will state this ~30–40% crossover as a
design rule.

## Reviewer #707B
> *W1: light on ADC physical-implementation details; the precise "20%" should be taken with a grain of salt.*
See **CR-4**: we specify 45 nm CMOS, MLSA-type ADC, 128×128 RRAM, 1-bit DAC, and
bit-sliced INT8, with $A_{ADC}\propto M\cdot2^{b}$ (Eq. 1). All areas are
**NeuroSIM-model estimates, not silicon**; the **relative** 20% saving is set by the
$2^{b}$ law and the per-group column counts, so it is robust to fine layout details.

> *C1 (comment): formulate as a "bit-reduction" benefit rather than an "area" benefit.*
Adopted — see **CR-4**: lead with **per-layer ADC bit-reduction**, presenting area
as the NeuroSIM consequence.

> *Q1: provide ADC-circuit/CIM assumptions, or stick to bit-number savings?*
We do **both** (see **CR-4**): assumptions are stated explicitly and we foreground
the bit-reduction result; the relative ADC-area saving follows from the $2^{b}$ law.

## Reviewer #707C
> *W1: PPL is coarse; it does not isolate the error source.*
PPL is used **deliberately** as the deployment metric, but we isolate the relevant
part: our signal is the **ADC-induced ΔPPL on a fixed quantization floor** (the floor
is orthogonal to ADC allocation; CR-2). We also decompose the ADC error itself
(Table S4, CR-1): the bit-reduction error is **decoupled** from the loss by
propagation, so no decomposed local proxy predicts it — we optimize the
directly-measured ΔPPL.

> *W2: evidence mostly group-level PPL; mechanism qualitative; ranking model-dependent (Qwen2 $W_{fc1}$).*
The mechanism is no longer only qualitative — CR-1/Table S4 give a verified
decomposition plus a measured local-error/loss decoupling that explains the proxy
failures — and it is **not load-bearing** (allocation consumes only measured ΔPPL). The
model-dependence is **architectural** (Qwen2's gated MLP makes $W_{fc1}$ the
sensitive FFN sub-layer); the architecture-independent claim **FFN>attention**
holds on all four models (Table 4) and probes/seeds (Table S2).

> *W3: validation limited (full allocation mainly OPT); gain over proxies modest.*
OPT-125M gives the full PPA+downstream pipeline and OPT-1.3B (145 layers) the full
allocation **at scale**; Pythia-410M and Qwen2-7B add architecture diversity (Qwen2
full PPA + downstream now in Table S3). The gain is modest because the contribution
is the **signal**: proxies protect the wrong layers (random costs +2.6%±1.4% at the
same budget; details in C5/C6).

> *C1: define "sensitivity" precisely.*
Agreed; the camera-ready adds a formal definition — sensitivity of group *g* = the
**test-PPL increase from reducing only group *g*'s ADC precision by one bit** under
the fixed CIM setup — a CIM-ADC/PPL-specific quantity, **not** general layer
importance or task sensitivity.

> *C2: justify PPL vs token-level NLL.*
See **Q1** (Table S1): the ranking and ILP allocation are identical under NLL (Spearman ρ=1.0).

> *C3: add mechanism diagnostics (MSE/SQNR, residual-norm, KL, drift).*
Done — **Table S4** reports granular/clipping MSE, SQNR, hidden-state drift, and
output KL per group, verifying the decomposition and the local-error/loss decoupling
(**CR-1**).

> *C4: separate model-specific from general ranking.*
See **W2**/**Q2**.

> *C5/C6: improvement is modest; clarify the contribution.*
Per **CR-1**, the contribution is that standard proxies protect the **wrong**
layers (saturation ρ=−0.70, Hessian ρ=0.20), so the **signal, not the solver**,
drives quality (random costs +2.6%±1.4%). The 30% Hessian closeness on OPT-125M is
coincidental — at 20.5% it is clearly worse (309.7 vs 308.6); at scale (OPT-1.3B)
measured sensitivity beats the saturation proxy at **both** budgets (−0.06% vs
+0.43%; +0.31% vs +0.89%, paper Table 4).

> *C7: combine with GPTQ/AWQ/SmoothQuant; operating-point realism.*
See **CR-2** (composes with weight-PTQ; ordering validated at usable OPT max-clip
and Qwen2 points).

> *C8: full pipeline on a modern non-OPT model.*
See **CR-3** and **Table S3**: the full profiling→ILP→PPA pipeline now runs on
Qwen2-7B at a usable operating point (34% ADC area+energy saving at +1.8% PPL;
ADC-read latency likewise reduced, Sec. 5.3; PIQA/BoolQ within ~1.6 pp of CIM-8b).

> *Q1: recompute sensitivity with token-level NLL / cross-entropy?*
**Table S1**: the per-group ranking is identical (**Spearman ρ = 1.0**) and the
group-level ILP allocation is unchanged. Since PPL = exp(mean NLL) is strictly
monotonic with a shared baseline, this invariance is provable; the additive
linear surrogate is therefore well-posed under the more additive NLL loss.

> *Q2: stronger mechanism evidence; why the top group differs OPT vs Qwen2.*
See **CR-1** and **Table S4** (analytical decomposition + diagnostics). The OPT-vs-Qwen2 difference is
**architectural, not contradictory**: Qwen2's gated MLP (SwiGLU) gives $W_{fc1}$ a
different role; the robust, architecture-independent claim is **FFN > attention**
(all four models, paper Table 4), which we separate from the model-specific
identity of the single most-sensitive FFN sub-layer.

## Reviewer #707D
> *W1: baseline PPL (~36 → ~300) is dominated by weight quantization.*
The high PPL is a fixed CIM quantization floor, **orthogonal** to our per-layer ADC
bit *allocation*; at usable operating points the ordering still holds (see **CR-2**).

> *W2: control/routing/mux/timing overheads of mixed-precision.*
See **CR-4**: group-level uniformity avoids per-column multiplexing and irregular
routing; the sole added element is a per-array config register (negligible area,
power, latency), included in the NeuroSIM estimate.

> *W3: mechanism is qualitative; the ILP uses a simplified linear degradation model.*
Mechanism: now analytical — see **CR-1** and **Table S4** (verified decomposition +
local-error/loss decoupling). The linear surrogate is **validated, not assumed**:
exhaustive group-level enumeration shows the ILP matches the brute-force optimum
(**regret = 0**, Sec. 4.2 / Table 2), and at moderate budgets the ILP reduces only
1–2 groups, where the per-group measurements are exact by construction.

> *C1: validate on a stronger, practically-usable quantized baseline.*
See **CR-2**: the ordering holds at usable points — OPT-125M max-clip 8b (PPL 93)
and Qwen2-7B (PPL 15.6 vs clean ~10, Table S3).

> *C2: include mixed-precision HW overhead (control, reconfig, mux, routing) in area/power/latency.*
See **W2**/**CR-4**.

> *C3: more rigorous ADC-noise propagation analysis.*
See **CR-1** and **Table S4**: we measure the ADC error decomposition and its
propagation — a ≈740× spread in local granular error collapses to ≈1.6× in
hidden-state drift, showing the loss impact is set by propagation, not local error.

> *Q1: does the ordering persist when weight-quant error is reduced?*
Yes — see **CR-2** (usable OPT max-clip and Qwen2 points; INT8 per-channel, Table S1).

> *Q2: timing mismatch / pipeline stalls with heterogeneous ADC bit-widths?*
See **CR-4**: group-level allocation keeps each array internally uniform and
column reads synchronous, so no per-column timing skew arises; the only overhead
is a per-array config register, included in the NeuroSIM area.

> *Q3: do benefits remain under non-exponential ADC area scaling?*
See **CR-4**: the method and ordering are unchanged; only the magnitude of the
saving depends on the ADC's area-vs-bit curve (largest for flash/MLSA, smaller
for SAR).

---

We thank the reviewers. The supplementary results (Tables S1–S4) and
clarifications address the central concerns — proxy failure and its mechanism, the
FFN>attention ordering and its robustness, metric invariance, weight-quant
independence, and ADC modeling — within the existing contribution.
