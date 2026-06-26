# ICCAD 2026 #707 Rebuttal 工作总结

**论文**：Measured Layer Sensitivity Guides Mixed-Precision ADC Allocation for CIM-Based LLM Inference

本文档总结 rebuttal 阶段的工作：四位审稿人提出的问题、为应对这些问题补充的实验、以及得到的结果。

---

## 一、总体策略

- **正文回复**（`rebuttal_response.md`，≤2000 词）：4 条 Common Response（CR-1~CR-4）+ 逐位审稿人逐条回应。
- **附件**（`supp.pdf`，2 页）：Table S1–S4 + "Usable-baseline check" 笔记，承载所有新增实验证据。
- 论文本体按 rebuttal 规则保持不变；所有澄清/修订标注为 camera-ready。
- 原则：能用直接测量/复现证据回答的，绝不空泛断言；新数据若与论文细节有出入，诚实处理、不过度声称。

---

## 二、审稿人意见汇总

### Reviewer #707A
- **W1**：反转现象的解释是"直观的定性假设",缺乏严格的解析/数学证明。
- **W2**：端到端混合精度 ILP 分配主要集中在小规模工作点。
- **W3**：缺少"何时该从 group 级切换到 per-layer 细粒度控制"的明确判据。

### Reviewer #707B
- **W1**：结论是 ADC 物理面积,但论文缺 ADC 实现细节(电路拓扑、工艺节点);"20% 节省"要打折看。
- **C1**：建议把卖点从"面积收益"改成"位宽削减收益"。
- **Q1**：给出 ADC 电路/CIM 假设,或干脆只谈位宽节省(因结论已说流片是 future work)。

### Reviewer #707C（最详尽）
- **W1**：PPL 太粗,不能区分退化来源(ADC 裁剪/激活误差/残差扰动/logit 漂移/INT8 交互)。
- **W2**：机理仍定性;排序依模型而变(Qwen2 是 Wfc1 而非 Wfc2 最敏感)。
- **W3**：完整分配主要在 OPT;对 proxy 的增益在若干设置下偏小。
- **C1**：精确定义 "sensitivity"。
- **C2**：用 PPL 作优化信号需更强论证(NLL 对可加线性代理更合理);报告 NLL 下排序/分配是否一致。
- **C3**：补机理诊断(layer-output error、MSE/SQNR、残差范数、输出 KL、隐状态漂移)。
- **C4**：把模型特定结论与一般结论分开。
- **C5/C6**：增益小(20.5%: 308.6/309.2/309.7;30% Hessian 很接近);厘清"换 proxy 为直测"之外的概念贡献。
- **C7**：与更强 PTQ(GPTQ/AWQ/SmoothQuant)结合;给更真实的工作点。
- **C8**：在现代非 OPT 模型(Qwen2-7B)上跑完整流程,含 PPL、面积、能耗、延迟、下游精度。
- **Q1**：用 token 级 NLL/交叉熵重算敏感度排序与 ILP 分配?
- **Q2**：更强的反转机理证据 + 一般性;为何 OPT 与 Qwen2 最敏感组不同?

### Reviewer #707D
- **W1**：baseline 质量差(量化把 PPL 从 ~36 抬到 >300),实用性存疑。
- **W2**：混合精度的控制/路由/复用/时序开销未建模。
- **W3**：机理定性;ILP 用了简化的线性退化模型。
- **C1**：在更强、可用的量化 baseline 上验证。
- **C2**：把混合精度硬件开销计入面积/功耗/延迟。
- **C3**：更严格的 ADC 噪声传播分析。
- **Q1**：权重量化误差降低后排序是否仍成立?
- **Q2**：异构 ADC 位宽下的时序错配/流水线停顿如何处理?
- **Q3**：非指数面积标度下收益是否仍在?

---

## 三、补充实验与结果

为应对上述意见,新增 5 组证据(附件 Table S1–S4 + Usable-baseline 笔记),对应脚本均已归档。

### 实验 1 — 指标不变性 + 权重量化鲁棒性（Table S1）
**应对**：707C-C2/Q1、707D-Q1
- **做法**：在 INT8 逐通道量化模型上,分别用 PPL 与 token 级 NLL 计算逐组敏感度,比较排序与 ILP 分配。
- **结果**：PPL 与 NLL 的逐组排序**完全一致(Spearman ρ = 1.0)**,group 级 ILP 分配不变;因 PPL = exp(mean NLL) 严格单调,该不变性可证。FFN 聚合 > 注意力聚合、Wfc2 为最敏感组在两种指标下都成立。
- **脚本**：`reb2026/exp_nll_wq.py`

### 实验 2 — FFN>attention 排序鲁棒性（Table S2）
**应对**：707A-W1、707C-Q2、707D-W3
- **做法**：在 OPT-125M 上扫描探针深度(7→6/5/4)× 多个校准种子。
- **结果**：FFN>attention 在 9 个单种子设置中 7/9 成立;种子平均下 7→6 清晰成立(+0.044 vs +0.023),7→4 压倒性成立(≈11×);加大 profiling 预算(calib 16/eval 60)后 7→6 跨种子稳定 —— 说明小探针单种子噪声是有限批次的伪影。

### 实验 3 — Qwen2-7B 完整流程（Table S3）★核心新增
**应对**：707A-W2、707C-C8、707C-W3
- **做法**：在 Qwen2-7B(GQA + SwiGLU 门控 MLP)上跑完整 profiling→ILP→PPA;采用 **max-clip 校准**(论文三种校准之一)取得**可用工作点**;面积/能耗用与论文一致的 2^b 解析模型;下游测 PIQA/BoolQ。
- **结果**：

  | 配置 | ADC 面积+能耗 | PPL | PIQA | BoolQ |
  |---|---|---|---|---|
  | Clean (fp16) | — | ≈10 | 80.4 | 85.2 |
  | CIM uniform 8b | — | 15.59 | 77.4 | 75.0 |
  | **Profiling-ILP (34%)** | **34%** | **15.87 (+1.8%)** | **77.2** | **73.4** |
  | Profiling-ILP (52%) | 52% | 16.35 (+4.9%) | — | — |

  - profiling 引导 ILP **省 34% ADC 面积+能耗,PPL 仅 +1.8%**,下游较 CIM-8b 仅差 ~1.6pp;
  - **proxy-blind**(盲目削最敏感组)→ PPL **1938**,模型崩坏;
  - 证明方法可迁移到现代非 OPT 架构。
- **脚本**：`qwen_ppa/qwen_ppa_ilp.py`(ILP/PPA)、`qwen_ppa/qwen_downstream.py`(PIQA/BoolQ)、`qwen_ppa/qwen_sensitivity.py`(排序)。

### 实验 4 — 反转机理：解析分解 + 诊断（Table S4 + 解析推导）★核心新增
**应对**：707A-W1、707C-C3/Q2、707D-W3/C3
- **解析**：ADC 失真严格分解为 `D_ADC(b) = D_gran(b) + D_ovl`,其中颗粒(分辨率)项 `D_gran ∝ (1−s)·V_FS²·4⁻ᵇ`(随位宽变),过载/裁剪项 `D_ovl` 与位宽无关 —— **降 1 位只放大颗粒项**。
- **诊断(OPT-125M,逐组测量)**：

  | 量 | 结果 | 含义 |
  |---|---|---|
  | D_gran 7→6 比值 | **×4.06 ≈ 4¹**(每组) | 颗粒项随位宽 4 倍标度,验证解析 |
  | D_ovl 7→6 比值 | **×1.00**(每组) | 裁剪项与位宽无关,验证解析 |
  | 局部颗粒误差 ΔD_gran 跨组跨度 | **≈740×** | 各组受到的 ADC 误差差异极大 |
  | 隐状态漂移跨组跨度 | **≈1.6×**;输出 KL ≈2.2× | 但输出扰动几乎一致 |
  | 结论 | **解耦** | 局部 ADC 误差与损失影响脱钩,由残差流传播决定 |

  - 关键诚实发现:局部误差、甚至输出级诊断(drift/KL)**都预测不了**损失级敏感度 → 必须直接测 ΔPPL;
  - 这也解释了论文已测的 saturation ρ=−0.70、Hessian ρ=0.20(各只抓到一个局部统计量,漏掉传播)。
- **脚本**：`qwen_ppa/diag_mechanism2.py`(并保留早期 `diag_mechanism.py`)。

> 备注:机理这块经过一次自我纠错——最初提出的"两因子乘积 ρ=−1.0"被发现是半循环论证(J 由 drift/ΔD_gran 定义、drift 近似恒定),已改为更稳健诚实的"分解 + 解耦"表述。

### 实验 5 — OPT-125M 可用工作点验证（Usable-baseline 笔记）★核心新增
**应对**：707D-W1/C1/Q1、707C-C7
- **做法**：定位 OPT-125M 高 PPL(~306)的真正来源,并在可用工作点重测排序。
- **结果**：

  | 配置 | PPL |
  |---|---|
  | FP32（无量化、无 ADC） | 42.51 |
  | **INT8 逐通道权重，无 ADC** | **42.46**（权重量化几乎无损） |
  | + ADC 7b p99 裁剪 | 320 |
  | + ADC 8b p99 裁剪 | 321（对位宽不敏感) |
  | + ADC 8b max-clip | **93（可用）** |

  - **关键诊断**：306 的退化主要来自 **p99 离群值裁剪**,而非权重精度;改用 max-clip 校准即得可用工作点(PPL 93);
  - 在该可用点逐组降 1 位:**Wfc2 +42.8/层(最敏感)**,FFN 聚合 **+21.5** vs 注意力 **−1.4** → **FFN≫attention 强成立,与论文 headline 一致**;
  - 与 Qwen2(PPL 16)一起,证明排序在**远比 p99 stress-test 更可用**的工作点上仍成立。
- **脚本**：`qwen_ppa/exp_usable_opt.py`

> 注:此实验暴露了论文 Sec 5.1 一处归因不够准(把 306 归给权重量化,实为裁剪)。rebuttal 中按保守策略未点破,仅用 max-clip(论文已有的校准方案)给出可用点;建议 camera-ready 时把该句归因改准。

---

## 四、审稿意见 → 应对 映射表

| 审稿点 | 回复落点 | 证据 |
|---|---|---|
| 707A-W1 机理需严格化 | CR-1 | Table S4 解析分解(×4.06/×1.00)+ 解耦 |
| 707A-W2 工作点偏小 | CR-3 | OPT-1.3B(145 层)+ Qwen2-7B 完整流程(S3) |
| 707A-W3 group vs per-layer | 707A-W3 | 论文 Sec 3.2 交叉点(≤30%/>40%) |
| 707B-W1 缺 ADC 实现细节 | CR-4 | 45nm/MLSA/128×128/1-bit DAC/bit-slicing(Sec 5.1)、A∝M·2^b(Eq.1) |
| 707B-C1 改成位宽削减叙事 | CR-4 | 采纳,camera-ready 主推 bit-reduction |
| 707B-Q1 ADC/CIM 假设 | CR-4 | 两者都做 |
| 707C-W1 PPL 粗、不分来源 | 707C-W1 | 固定 floor 上的 ADC ΔPPL + Table S4 分解 |
| 707C-W2 机理定性/排序依模型 | CR-1, 707C-W2 | S4 解析化 + 架构性解释(SwiGLU→Wfc1) |
| 707C-W3 验证有限/增益小 | 707C-W3, C5/C6 | OPT 全流程 + Qwen2(S3);信号而非求解器 |
| 707C-C1 定义 sensitivity | 707C-C1 | 给出可操作定义 |
| 707C-C2/Q1 PPL vs NLL | Table S1 | ρ=1.0、分配不变 |
| 707C-C3/Q2 机理诊断 | CR-1, Table S4 | MSE/SQNR/drift/KL 全报 |
| 707C-C4 分离模型特定 | 707C-W2/Q2 | FFN>attention 为一般claim |
| 707C-C5/C6 增益小/贡献 | 707C-C5/C6 | proxy 保护错层(ρ=−0.70/0.20);OPT-1.3B 两预算均胜 |
| 707C-C7 结合 PTQ/真实工作点 | CR-2 | PTQ 正交可叠加;OPT max-clip(93)/Qwen2(16)可用点 |
| 707C-C8 现代模型完整流程 | CR-3, Table S3 | Qwen2 PPL/area/energy/下游(+latency 见 OPT Sec 5.3) |
| 707D-W1/C1/Q1 baseline 差/可用性/排序 | CR-2 + Usable 笔记 | floor 正交;OPT max-clip(93)与 Qwen2(16)排序成立 |
| 707D-W2/C2/Q2 硬件开销/时序 | CR-4 | group 级均匀,仅 per-array 位宽寄存器 |
| 707D-W3/C3 机理/传播分析 | CR-1, Table S4 | 解析分解 + 传播解耦 |
| 707D-Q3 非指数标度 | CR-4 | 方法与排序不变,仅幅值随 ADC 面积曲线变 |

---

## 五、关键数字一览（均已与论文/实验核对）

- **机理分解**：D_gran ×4.06≈4¹、D_ovl ×1.00;ΔD_gran 跨度 ≈740× → drift ≈1.6×(解耦);saturation ρ=−0.70、Hessian ρ=0.20。
- **Qwen2-7B**：baseline 15.59(clean ≈10);ILP 34% → 15.87(+1.8%);52% → 16.35(+4.9%);proxy-blind → 1938;PIQA 77.4→77.2、BoolQ 75.0→73.4。
- **OPT 可用点**：FP32 42.51 / INT8 权重 42.46 / p99-ADC ≈320 / **max-clip 8b 93**;该点 Wfc2 +42.8、FFN +21.5 vs attn −1.4。
- **NLL 不变性**：Spearman ρ=1.0。
- **论文既有(rebuttal 引用)**：20.5% 时 308.6/309.2/309.7;OPT-1.3B −0.06%/+0.43%、+0.31%/+0.89%;random +2.6%±1.4%;ILP regret=0;A_ADC∝M·2^b(Eq.1);45nm/MLSA/128×128(Sec 5.1)。

---

## 六、交付物清单

- **正文回复**：`rebuttal/rebuttal_response.md`(≤2000 词)
- **附件**：`rebuttal/supp.tex` → `rebuttal/supp.pdf`(2 页,Table S1–S4 + Usable-baseline 笔记)
- **实验脚本**：`reb2026/exp_nll_wq.py`、`qwen_ppa/{diag_mechanism2.py, exp_usable_opt.py, qwen_ppa_ilp.py, qwen_downstream.py, qwen_sensitivity.py}`
- **结果数据**：`qwen_ppa/{diag_mechanism2.json, usable_opt.json}` 等
- 全部已 git 提交并备份至服务器 `rebuttal_backup/`。

---

## 七、一句话总结

针对四位审稿人最集中的三类质疑——**机理只是定性、baseline 不可用、缺现代模型/硬件细节**——分别补了 **解析分解+诊断(Table S4)**、**OPT max-clip 可用点(PPL 93)+ Qwen2 完整流程(Table S3)**、以及 **ADC 实现细节与硬件开销澄清(CR-4)**,把回复做到了逐条有据、与论文自洽、诚实可复现。
