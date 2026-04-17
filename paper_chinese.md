# 实测层敏感度指导存算一体 LLM 推理中的混合精度 ADC 分配

> **会议**：ICCAD 2026 — IEEE/ACM 计算机辅助设计国际会议，2026年11月8-12日，美国加州圣荷西
>
> **原标题**：Measured Layer Sensitivity Guides Mixed-Precision ADC Allocation for CIM-Based LLM Inference

---

## 摘要

存算一体（Compute-in-Memory, CIM）架构通过在阻变存储阵列中直接执行 Transformer 层的矩阵向量乘法来加速大语言模型（LLM）推理，但其模数转换器（ADC）占据了芯片面积的主导部分。混合精度 ADC 分配能够降低这一成本，然而传统的代理信号——饱和率与 Hessian 迹——无法可靠地预测哪些 Transformer 层在 CIM 场景下能够容忍更低的 ADC 精度。

我们直接分析每类层在 ADC 位宽降低条件下的敏感度，揭示了一种**饱和-敏感性反转**现象：饱和率最高的注意力层（$W_{qkv}$）是精度敏感度**最低**的层，而低饱和率的前馈网络层（$W_{fc2}$）反而是最敏感的——与传统代理指标给出的方向恰好相反。

基于此发现，我们提出一种低成本的分析引导分配流程：仅需 $G+1$ 次评估即可提取敏感度信号，并将其输入面积感知的整数线性规划（ILP）求解器。在 OPT-125M 上，该流程在相同 ILP 求解器下优于饱和率引导、Hessian 引导和随机基线方法，以不到 1% 的相对困惑度（PPL）退化实现 20% 的 ADC 面积节省。在 OPT-1.3B 上，在两个测试预算下均优于饱和率引导分配。迁移验证实验表明，该敏感度排序可泛化至更高精度的部署场景，且精度代价可忽略不计。

**关键词**：存算一体、大语言模型、模数转换器、混合精度分配、敏感度分析

---

## 1 引言

> **[图1]**（fig_methodology）：提出的分配流程。(1) CIM 仿真评估 LLM 各层，支持逐层可配置的 ADC 精度。(2) 直接的组级分析，通过每次降低一组来测量各组的 ΔPPL。(3) 面积感知的 ILP 在 ADC 面积预算下分配 ADC 位宽，将低敏感度的 $W_{qkv}$ 层降至 6-bit，同时保持敏感的 FFN/输出组在 7-bit。(4) NeuroSIM 验证最终方案的精度、ADC 面积、能耗和延迟权衡。

CIM 架构 [1,2] 通过在阻变存储阵列中直接执行矩阵向量乘法来加速 LLM 推理。负责将累积列电流数字化的 ADC，其面积随位宽呈指数增长：NeuroSIM [3] 报告，对于 OPT-125M [4]，ADC 在 7-bit 时占芯片总面积的 24.4%，在 10-bit 时增长至 63.9%。混合精度 ADC 分配——为容错层分配更少位宽——可以显著降低该成本。核心挑战在于如何识别哪些层能够容忍精度降低。

现有的数字混合精度方法 [5,6] 依赖 Hessian 迹或饱和率来指导位宽分配。然而，在我们研究的 CIM ADC 场景下，这些代理指标未能预测实测敏感度：饱和率与实际精度影响呈**负相关**，Hessian 迹则**无相关性**（详见第5节）。因此，基于传统饱和率引导的策略会**保护错误的层**。

我们通过直接测量各类层在 ADC 位宽降低条件下的敏感度来弥补这一差距，揭示了**饱和-敏感性反转**：饱和率最高的层（$W_{qkv}$）是最不敏感的，而饱和率最低的层（$W_{fc2}$）反而最敏感。该信号获取成本低（$G+1$ 次评估即可），且具有可迁移性——低精度分析所得的组级 ILP 分配方案在部署于更高精度时保持一致。利用该信号，所提出的分配流程在相同面积预算下优于所有测试的代理基线方法（图1）。

**贡献：**

1. 我们证明饱和率和 Hessian 代理指标在 CIM 场景下无法预测 ADC 敏感度，而直接分析揭示了相反的排名——在 OPT-125M/1.3B 上得到验证，并在 Pythia-410M 和 Qwen2-7B 上确认了方向一致性。

2. 我们提出一种组级分析协议（算法1），仅需 $G+1$ 次评估即可提取分配信号。所得排序在四个工作点上保持稳定，并可无精度损失地迁移至更高精度部署。

3. 在相同 ILP 求解器下，实测敏感度在 OPT-125M（三种基线对比）和 OPT-1.3B（饱和率基线对比）上始终产生更优的分配方案，包括在实际高精度部署场景中。

---

## 2 背景与相关工作

### 2.1 CIM 阵列架构与 ADC 瓶颈

> **[图2]**（fig_overview_cim）：LLM 到 CIM 的映射与 ADC 面积动因。(a) 一个典型的 decoder-only LLM block 被分解为 CIM 映射的线性组，包括 $W_{qkv}$、$W_{out}$、$W_{fc1}$、$W_{fc2}$ 和 $W_{head}$，即本文分析的敏感度组。分拆式和融合式 QKV 实现均由 $W_{qkv}$ 代表，门控 MLP 投影映射至相应 FFN 组。(b) 线性投影映射至 CIM 子阵列，模拟列电流累加执行矩阵向量乘法，ADC 对输出进行数字化。(c) ADC 面积随列数和位宽呈指数增长（$A_{ADC} \propto M \cdot 2^b$），使得均匀高精度 ADC 代价高昂，由此激发混合精度 ADC 分配。

图2(a) 将 decoder-only LLM 的线性投影按功能分为本文使用的敏感度组，图2(b) 展示了它们到 CIM 子阵列的映射。在 CIM crossbar 中，权重 $\mathbf{W}$ 被编程为阻变存储单元的电导状态。输入向量 $\mathbf{x}$ 通过 DAC 按行施加，列电流根据基尔霍夫电流定律累加后由 ADC 数字化。

如图2(c) 所示，对于具有 $M$ 列的子阵列配备 $b$-bit MLSA 型 ADC [3]，ADC 面积按如下公式缩放：

$$A_{ADC} \propto M \cdot 2^b \tag{1}$$

因为多级灵敏放大器需要 $2^b$ 个比较器电平。在 OPT-125M 的全部 73 个线性层（4 种不同的权重矩阵形状）上，此指数增长产生了图3所示的面积分布——ADC 占比从 7-bit 的 24% 增长至 10-bit 的 64%。

> **[图3]**（fig_ppa_curve）：(a) 芯片面积 vs. ADC 精度（NeuroSIM）：$2^b$ 缩放导致从 7b 到 10b 增长 3 倍。(b) ADC 占芯片总面积的比例：在 9-bit 以上 ADC 占比超过 50%。

### 2.2 激活异常值与 ADC 饱和

LLM 的激活值呈现系统性的**异常通道**现象 [7]：某些特征维度的幅值远超中位数。如图4所示，不同层类型的饱和率差异巨大——$W_{qkv}$/$W_{fc1}$ 接近 100% 饱和，而 $W_{out}$/$W_{fc2}$ 保持在 20% 以下——且该模式从 OPT-125M 到 OPT-1.3B 持续存在。

> **[图4]**（fig2_saturation_heterogeneity）：7-bit 下逐层的 max-clip ADC 饱和率。(a) OPT-125M（73层）：$W_{qkv}$/$W_{fc1}$ 接近 100% 饱和。(b) OPT-1.3B（145层）：相同模式持续存在。两个模型中 $W_{out}$/$W_{fc2}$ 均低于 20%。

一个被广泛使用的层难度代理指标是 **max-clip 饱和率**：

$$s_i^{max} = \Pr\left[|y_{ij}| > V_{FS}^{max}\right] \tag{2}$$

该代理指标（公式2）认为具有高 $s_i^{max}$ 的层需要更多 ADC 位宽。SmoothQuant [8] 通过将激活难度迁移至权重来缓解异常值。然而正如我们在第5节所展示的，$s_i^{max}$ 并不能可靠预测降低 ADC 精度的实际精度影响。

### 2.3 相关工作

**面向 LLM 的 CIM。** PRIME [1] 和 ISAAC [2] 率先将 CIM 应用于 CNN；近期综述 [9] 指出 ADC 是 CIM 扩展至 LLM 时的瓶颈。FlexCiM [10] 和 P3-LLM [11] 针对 LLM CIM 推理，但未优化逐层 ADC 精度。

**ADC 感知的 CIM。** Azamat 等 [12] 和 CIM²PQ [13] 将混合精度 ADC 应用于 CNN 加速器。Chen 等 [14] 采用基于 Hessian 的敏感度分析用于 ReRAM CIM；但在我们的 LLM ADC 场景下，Hessian 迹与实测敏感度无相关性（$\rho=0.20$），饱和率则呈负相关（$\rho=-0.70$）。Wan 等 [15] 和 MXFormer [16] 在处理器级别验证了混合精度 CIM。

**LLM 量化。** SmoothQuant [8]、GPTQ [17] 和 AWQ [18] 面向数字加速器。HAWQ [5] 和 HAQ [6] 使用 Hessian 迹或强化学习进行数字量化；这些代理指标不适用于 CIM ADC 噪声。Yan 等 [19] 通过训练时噪声注入解决 NVCiM 最坏情况精度问题；我们的工作通过部署时分配与其互补。我们使用 DNN+NeuroSIM [3,20,21] 进行电路级验证。

---

## 3 敏感度分析方法

### 3.1 测量协议

我们将层组 $i$ 的**实测敏感度**定义为降低其 ADC 精度所引起的困惑度增量：

$$\sigma_i = \frac{PPL(\mathbf{b}^{(i)}) - PPL(\mathbf{b}^{ref})}{|g_i|} \tag{3}$$

其中 $\mathbf{b}^{ref} = [b_{nom}, \ldots, b_{nom}]$ 是均匀基线，$\mathbf{b}^{(i)}$ 仅将组 $i$ 降至 $b_{prb}$，其他层保持 $b_{nom}$ 不变。

图5展示了我们的定性假说：不同线性组在 decoder block 中引发不同的噪声传播路径。

> **[图5]**（fig_noise_path）：饱和-敏感性反转的定性噪声传播假说。注入 $W_{qkv}$ 的噪声经过注意力分数计算、softmax/多头混合、$W_{out}$ 和残差/归一化路径后才影响后续状态，与 OPT-125M 7b→6b 分析中较低的实测敏感度一致。相反，注入 $W_{fc2}$ 的噪声在 FFN 之后直接进入残差流，可影响后续 decoder block，与较高的实测敏感度一致。该图展示的是对实测趋势的解释，而非形式化证明。

为提高效率，我们采用组级测量（公式3）——同时降低同一功能类型的所有层——从而只需 $G=5$ 次测量而非 73 次。算法1形式化了该协议。

---

**算法1：组级 ADC 敏感度测量**

**输入**：模型 $\mathcal{M}$，校准数据 $\mathcal{D}_{cal}$，分析数据 $\mathcal{D}_{prof}$，标称位宽 $b_{nom}$，探测位宽 $b_{prb}$

**输出**：各组实测敏感度 $\{\sigma_g\}_{g=1}^G$

1. 将 $N$ 层按功能类型划分为 $G$ 组
2. 使用 $\mathcal{D}_{cal}$ 为每层 $i$ 校准 ADC 满量程 $V_{FS,i}$
3. $\mathbf{b}^{ref} \leftarrow [b_{nom}] \times N$
4. $PPL_{ref} \leftarrow Eval(\mathcal{M}, \mathbf{b}^{ref}, \mathcal{D}_{prof})$
5. **对每组** $g = 1, \ldots, G$：
   - $\mathbf{b}^{(g)} \leftarrow \mathbf{b}^{ref}$；将组 $g$ 中所有 $i$ 设为 $b_i^{(g)} = b_{prb}$
   - $PPL_g \leftarrow Eval(\mathcal{M}, \mathbf{b}^{(g)}, \mathcal{D}_{prof})$
   - $\sigma_g \leftarrow (PPL_g - PPL_{ref}) / |g|$
6. **返回** $\{\sigma_g\}$

---

### 3.2 组级粒度验证

我们通过深度分区分析和三个代表性 block 的逐层抽查来验证组级分析的有效性（图6）。

> **[图6]**（fig_grouping_validity）：组级分析有效性。(a) 各层类型的逐深度敏感度：所有 $W_{qkv}$ 分区为负值，所有 $W_{fc2}$ 分区为正值，在每个深度位置确认了组间排序。(b) 跨深度分区的均值±标准差。

FFN 层在每个深度位置上一致地比注意力层更敏感；敏感度总体随深度递减，但 $W_{fc2}$ 即使在 Late 位置仍保持 0.4 ΔPPL 以上。我们进一步对比了组级 ILP 与逐层 ILP（73 个独立敏感度测量）：在中等预算（≤30% 节省）下，组级 ILP 优于逐层 ILP，因为组级平均降低了有限评估批次带来的测量噪声。逐层 ILP 仅在激进预算（>40%）下恢复优势。

我们将 WikiText-2 [22] 评估集划分为三个不重叠的子集：
- **$\mathcal{D}_{cal}$**（4 批次）：用于 ADC 满量程校准
- **$\mathcal{D}_{prof}$**（10 批次，约 5K tokens）：用于敏感度分析（算法1）
- **$\mathcal{D}_{test}$**（100 批次，约 51K tokens）：用于表3所报告的所有最终 PPL 评估

所有序列长度为 512 tokens，采用 p99-clip ADC 校准。分析集与测试集不重叠；分配决策仅基于 $\mathcal{D}_{prof}$。

### 3.3 实测敏感度的鲁棒性

表1总结了五个维度上的鲁棒性验证。在此组级搜索空间中，ILP 分配方案在校准、工作点和迁移条件下保持不变；在增强噪声模型下，7 种配置中有 5 种保持相同分配方案（表1）。在已评估的 OPT 设置中，$W_{qkv}$ 始终是低敏感度组。

**表1：鲁棒性总结。**"分配是否稳定？"= ILP 在所有测试条件下是否产生相同的受保护组集合。

| 维度 | 条件数 | 稳定？ |
|------|--------|--------|
| 校准种子 | 5 个偏移 | 是 (5/5) |
| 校准方案 | p99 / p99.5 / max | 是 (3/3) |
| 工作点 | 7→6 / 8→7 / 9→8 / 10→9 | 是 (4/4) |
| 迁移 | Profile@7, deploy@8/9/10 | 是 (3/3) |
| 噪声模型 | ADC + 6 种热/器件变异 | 部分 (5/7) |

---

## 4 面积感知的位宽预算

我们将带预算约束的位宽分配问题建模为整数线性规划（ILP），将实测敏感度、层特定面积系数和位宽选择约束整合为单一优化步骤。

### 4.1 ILP 公式

每一层继承其功能组的实测敏感度：对所有 $i \in g$，$\sigma_i = \sigma_g$。给定 $N$ 层的敏感度和目标 ADC 面积预算 $B$，我们最小化总的敏感度加权退化：

$$\min_{\mathbf{b}} \sum_i \max(0, b_{nom} - b_i) \cdot \sigma_i^{+} \quad \text{s.t.} \; \sum_i a_i \cdot 2^{b_i} \leq B, \; b_i \in \mathcal{B} \tag{4}$$

其中：
- $\sigma_i^{+} = \max(\sigma_i, 0)$ 将负敏感度截断为零（负值可能源于评估噪声，若不截断则会导致求解器利用有限样本方差）
- $b_{nom}$ 为标称位宽（分析时为 7，部署时为 10）
- $\mathcal{B}$ 为允许的位宽集合（7b 压力测试场景为 $\{4,5,6,7\}$；10b 部署场景为 $\{7,8,9,10\}$；实际上在测试的中等预算下 ILP 仅选择 6 和 7）
- $a_i$ 为层特定的 ADC 面积系数，与 NeuroSIM 映射中的 ADC 列数成正比
- 面积约束遵循 MLSA 模型（公式1）

由此产生的混合精度 ADC 面积节省为：

$$\Delta A_{ADC} = \sum_i a_i \left(2^{b_{nom}} - 2^{b_i^*}\right) \tag{5}$$

我们通过二值指示变量 $x_{i,b} \in \{0,1\}$ 对公式(4)进行线性化，并使用 `scipy.optimize.milp` 求解。对于 OPT-125M（292 个二值变量），ILP 在 0.1 秒内求解至证明最优。对所有 $4^5 = 1024$ 种组级配置的穷举枚举证实 ILP 解与暴力搜索最优解一致。

### 4.2 替代模型验证

线性替代模型对逐对 ΔPPL 预测是近似的（表2、图7），但其主要作用是**分配排名**。两个强验证信号证实其正确性：(1) 穷举枚举验证 ILP 与暴力搜索最优解一致（regret = 0.0）；(2) 端到端实测 PPL 独立验证分配质量。该替代模型足够有效，因为中等预算 ILP 仅降低一到两个组，此时个体测量在构造上是精确的。

**表2：替代模型验证。** 上部：分配最优性（ILP 匹配全局最优）。下部：代表性逐对交互测试。

| | |
|---|---|
| **分配最优性（20.5% 节省）** | |
| ILP 分配 PPL | 308.6 (+0.7%) |
| 暴力搜索全局最优 PPL | 308.6 (+0.7%) |
| 分配遗憾（regret） | **0.0 PPL** |
| **逐对交互（代表性组合）** | 预测 Δ → 实测 Δ |
| $W_{out} + W_{fc2}$ | +11.4 → +11.7 |
| $W_{qkv} + W_{fc2}$ | +10.9 → +9.4 |
| $W_{fc1} + W_{head}$ | +9.4 → +10.7 |

> **[图7]**（fig_interaction_scatter）：替代模型可加性验证：预测 vs. 实测逐对 ΔPPL。绿色区域=高估（保守）；红色=低估。平均|误差| = 5.3 PPL，但 ILP 分配仍匹配暴力搜索全局最优（regret = 0.0）。

### 4.3 计算成本

敏感度测量需要 $G+1 = 6$ 次评估；每次评估在完整的 $\mathcal{D}_{prof}$ 子集上运行 10 批次（约 5K tokens）。ILP 包含 $N \times |\mathcal{B}|$ 个二值变量（OPT-125M 为 292 个，<0.1s；OPT-1.3B 为 580 个，<0.2s）。总分析开销相比部署可忽略不计。

---

## 5 实验评估

我们使用 NeuroSIM [3] 电路级分析（45nm CMOS、128×128 RRAM crossbar、MLSA 型 ADC、1-bit DAC 输入、INT8 权重映射 + bit-slicing）验证完整的分配流程，涵盖面积、能耗、延迟和下游任务精度。

### 5.1 设置与参考定义

我们定义四个参考工作点：
- **FP32**：无 CIM 噪声或权重量化
- **CIM-*b*b**：INT8 权重 + *b*-bit ADC，p99-clip 校准
- **分析场景**：CIM-7b→6b，敏感度对比最显著
- **部署场景**：CIM-8b/9b/10b，实际芯片运行的精度

FP32 到 CIM 的精度差距主要由未经微调的 INT8 权重量化主导，而非 ADC 噪声。为证实这一点，我们在与图8跨场景验证相同的 10 批次子集上，测量了仅 INT8 的控制实验（12-bit ADC，有效消除 ADC 噪声）以及 CIM-7b 和 CIM-10b；绝对 PPL 值因不同评估子集而与表3的 100 批次 $\mathcal{D}_{test}$ 不同，但在该匹配协议内的**相对**比较是有效的：

| 配置 | PPL |
|------|-----|
| FP32（无量化） | 36.2 |
| 仅 INT8（12b ADC） | 316.9 |
| INT8 + 10b ADC | 316.9 |
| INT8 + 7b ADC | 317.1 |

仅 INT8 与 CIM-7b 基线仅相差 0.2 PPL，表明 ADC 截断相比 INT8 权重量化差距的贡献可忽略。我们的目标不是恢复 INT8 量化损失，而是在固定的 CIM 量化底线之上降低 ADC 成本；改进的权重量化（如 GPTQ、AWQ）是正交的，可与我们的分配流程结合使用。

在一致的评估协议下，CIM-7b 到 CIM-10b 基线几乎相同（图8a）；该 10 批次跨场景扫描仅用于验证基线稳定性，所有最终分配比较均报告在表3的 100 批次 $\mathcal{D}_{test}$ 上。我们所解决的分配问题——在权重量化底线之上以最小的**额外**退化降低 ADC 面积——在实际中仍然有价值，因为不同分配方法之间的相对 PPL 差异是一致且可复现的，且面积节省在实际模型规模下可达数百 mm²。

> **[图8]**（fig_deployment_regime）：(a) CIM 基线 PPL 在 7-10b ADC 间几乎相同（差异 = 0.3 PPL；权重量化主导）；在四个场景下使用一致的 10 批次协议评估。(b) 迁移分配：7b→6b 分析在三个测试的更高精度场景下均产生与原生分析相同的组级 ILP 分配方案。该 10 批次扫描验证跨场景基线稳定性；所有最终分配比较使用表3的 100 批次 $\mathcal{D}_{test}$。

### 5.2 敏感性反转与跨场景迁移

实测敏感度（图9）揭示：饱和率将 $W_{qkv}$ 和 $W_{fc1}$ 排为最"危险"的层（约 100% 饱和），但实测敏感度将它们排为**最末**（7b→6b 下分别为 0.13 和 0.20 ΔPPL/层）。$W_{fc2}$ 以 1.41 ΔPPL/层排名第一，尽管其饱和率仅 19%——**11 倍的反转**。

> **[图9]**（fig_sensitivity_heatmap）：(a) 四个工作点下的实测敏感度（ΔPPL/层，OPT-125M）。$W_{fc2}$ 在 3/4 场景下最敏感；$W_{qkv}$ 始终最不敏感。(b) 饱和率代理 vs. 实测排名：代理会保护**错误**的层（$\rho = -0.70$）。

不同相邻场景间的精确排名相关性有所变化；对于非相邻和高精度场景对，Spearman $\rho$ 范围为 0.60 至 0.90。更重要的是，对分配至关重要的排序是稳定的：$W_{qkv}$ 始终位于低敏感度集合，FFN 层在每个测试工作点上始终比注意力层更敏感。

我们将此反转归因于定性的噪声传播效应，而非将其视为形式化证明（图5）。对于 $W_{qkv}$，ADC 扰动经过注意力分数计算、softmax/多头混合、$W_{out}$ 和残差/归一化路径后才影响后续隐藏状态，这些步骤可衰减其实测影响。对于 $W_{fc2}$，扰动在 FFN 之后直接注入残差流，可影响后续的 decoder block，与较高的实测敏感度以及图6(a) 中的中层深度敏感度峰值一致。

FFN > 注意力的层级关系在所有四个测试工作点上均成立（图9a）。图10可视化了排名稳定性和分配重叠：7b→6b 分析在三个测试的更高精度场景下均产生 5/5 的组级一致，组级分配遗憾为零（PPL 在基线 0.2% 以内）。

> **[图10]**（fig_transfer_heatmap）：(a) 四个工作点下的敏感度排名：$W_{qkv}$ 始终处于低敏感度集合，FFN > 注意力的排序全程保持。(b) 场景间的逐对 Spearman $\rho$（非相邻对为 0.60-0.90）。

### 5.3 分配结果

表3报告了所有配置的实测 PPL。为将实测敏感度**信号**的价值与 ILP **求解器**分离，我们将四种分配信号在相同预算下通过同一 ILP 运行。在 20.5% ADC 面积节省下，分析引导 ILP（+0.7%）优于 Hessian-ILP（+1.1%）、饱和率-ILP（+0.9%）和随机分配（+2.6%±1.4%，20 次试验）。

对于代理基线，我们通过对角 Fisher 近似在 $\mathcal{D}_{prof}$ 上使用相同的 10 个分析批次估计每组 Hessian 迹，跨组归一化后将所得代理分数输入相同的 ILP 求解器；饱和率-ILP 使用每组饱和率作为代理信号。相同模式在 30% 节省下持续存在。穷举组级暴力搜索证实 ILP 解与全局最优组级配置一致。图11可视化了逐层分配；图12展示了 PPL-面积 Pareto 前沿。

> **[图11]**（fig5_bit_assignment）：20.5% ADC 面积节省下的逐层位宽分配。(a) 实测-ILP 将 $W_{qkv}$（最不敏感）降至 6-bit。(b) 饱和率-ILP 将 $W_{out}$/$W_{fc2}$（最敏感）降至 6-bit——保护了**错误**的层。相同 ILP 求解器，不同信号，不同分配。

> **[图12]**（fig6_pareto_frontier）：PPL vs. ADC 面积 Pareto 前沿。Mixed-ILP（红色）在所有评估预算下均优于均匀位宽扫描（灰色）。SQ+6b 作为正交激活重缩放参考展示，而非 ADC 分配基线。

**表3：多种信号和预算下的分配结果（OPT-125M，7b 场景）。** 所有行使用 p99 校准和 100 批次 $\mathcal{D}_{test}$。Sav.% = 相对均匀 7b 的 ADC 面积节省；Rel. = 相对均匀 7b 的 PPL 退化；Area = 芯片总面积（mm²）。SQ+6b 为正交激活重缩放参考，非 ADC 分配基线。Random (20) 报告 20 次独立随机分配的均值±标准差。

| 信号 | 求解器 | Sav.% | PPL | Rel. | Area |
|------|--------|-------|-----|------|------|
| **均匀基线** | | | | | |
| --- | 7b | 0 | 306.4 | --- | 937 |
| --- | 6b | 50 | 315.3 | +2.9% | 822 |
| **20.5% 节省** | | | | | |
| **实测** | **ILP** | **20.5** | **308.6** | **+0.7%** | **890** |
| 饱和率 | ILP | 20.5 | 309.2 | +0.9% | 890 |
| Hessian | ILP | 20.5 | 309.7 | +1.1% | 890 |
| 随机 (20) | --- | 20.5 | 314.5±4.4 | +2.6%±1.4% | 890 |
| 组级全局最优 | BF | 20.5 | 308.6 | +0.7% | 890 |
| **30% 节省** | | | | | |
| 实测 | ILP | 30 | 309.3 | +0.9% | 861 |
| 饱和率 | ILP | 30 | 310.5 | +1.3% | 861 |
| Hessian | ILP | 30 | 309.5 | +1.0% | 861 |
| **正交参考** | | | | | |
| --- | SQ+6b | 50 | 305.0 | -0.5% | 822 |

在 10-bit 部署场景的单独验证中，CIM-7b 到 CIM-10b 基线产生几乎相同的 PPL（差异 <0.2%），证实在权重量化底线之上 ADC 精度的贡献可忽略。分析引导 ILP 在 10b 下节省 20% ADC 面积（通过公式5计算为 361 mm²），PPL 变化可忽略，因为 ILP 将 ADC 预算集中在敏感层上，同时降低不敏感的 $W_{qkv}$ 层。

在相同 NeuroSIM 映射下，20.5% ADC 面积节省方案将 NeuroSIM 报告的 **ADC 路径能耗降低 24.5%**，**ADC 读取延迟降低 12.5%**。

SmoothQuant [8] 是一种正交的激活预处理步骤；它在 ADC 转换前降低异常值幅度，可通过在 SQ 变换后的模型上重新分析敏感度来与分析引导分配结合使用。我们报告 SQ+6b 作为独立参考；SQ + 分析引导 ILP 的组合是未来工作的自然扩展。

代理排名分析（图13）进一步证实饱和率（$\rho = -0.70$）和 Hessian（$\rho = 0.20$）与实测敏感度呈负相关或无相关。

> **[图13]**（fig_proxy_scatter）：代理失效：饱和率 (a) 和 Hessian 迹 (b) vs. 实测敏感度。两种代理都会保护**错误**的层。

**下游任务精度。** 我们在五个零样本任务（每个 570-3000 个样本）上评估，并给出 95% bootstrap 置信区间（图14）。ILP-20% 在所有任务上均保持在 CIM-7b 的置信区间内（$|\overline{\Delta}| = 0.2$ 百分点）。

> **[图14]**（fig_downstream_ci）：零样本精度与 95% bootstrap 置信区间。ILP-20%（绿色）在每个任务上均落在 CIM-7b 的置信区间内。平均 $|\overline{\Delta}| = 0.2$ 百分点。

### 5.4 跨模型验证与噪声鲁棒性

表4报告了四个模型的敏感度排序检查以及 OPT-1.3B 的完整分配结果。OPT-1.3B 分配证实了规模化的实际价值：分析引导 ILP 在 20% 和 30% 节省下均优于饱和率-ILP。

**表4：跨模型验证。** 上部：四个模型的敏感度排序检查（Pythia/Qwen 仅报告排序）。下部：OPT-1.3B（145 层）的完整分配结果。

*敏感度排序：*

| 模型 | 架构 | 探测 | 最敏感 | 最不敏感 | FFN>Att? |
|------|------|------|--------|----------|----------|
| OPT-125M | OPT | 7→6 | $W_{fc2}$ | $W_{qkv}$ | 是 |
| OPT-1.3B | OPT | 7→6 | $W_{fc2}$ | $W_{qkv}$ | 是 |
| Pythia-410M | NeoX | 7→6 | FFN | 注意力 | 是 |
| Qwen2-7B | Qwen2 | 8→7 | $W_{fc1}$ | $W_{out}$ | 是 |

*OPT-1.3B 分配（7b 基线 = 672.6）：*

| 信号 / 预算 | PPL | Rel. |
|-------------|-----|------|
| **分析引导 / 20%** | **672.2** | **-0.06%** |
| 饱和率-ILP / 20% | 675.5 | +0.43% |
| 分析引导 / 30% | 674.7 | +0.31% |
| 饱和率-ILP / 30% | 678.6 | +0.89% |

我们进一步在 ADC 截断模型上叠加加性热噪声和乘性器件变异（六种配置）。在 7 种设置中，$W_{fc2}$ 在 5 种中保持为最敏感的非 $W_{head}$ 层（图15）；ILP 分配方案在 5/7 种配置中保持稳定（表1）。

> **[图15]**（fig_robustness_original）：六种增强 CIM 噪声配置下的敏感度鲁棒性（热噪声 + 器件变异）。$W_{fc2}$ 在 5/7 种设置中保持为最敏感的非 $W_{head}$ 层；在最强器件变异（$\sigma_{dev}=0.03$）下，$W_{fc2}$ 表现出最大正 ΔPPL（+2.86/层）。

---

## 6 结论

我们已经证明，在测试的 CIM 场景下，直接实测的层敏感度提供了比饱和率或 Hessian 代理更可靠的 ADC 分配信号。关键的经验发现——饱和-敏感性反转，即饱和率最高的层是精度敏感度**最低**的——可在 $G+1$ 次评估中提取，并在校准种子、工作点和迁移场景间保持稳定。

在 OPT-125M 上，分析引导 ILP 在相同求解器下优于饱和率引导、Hessian 引导和随机引导基线，以 +0.7% 的相对 PPL 实现 20% ADC 面积节省（表3）。在 OPT-1.3B 上，在两个测试预算下均优于饱和率引导分配。迁移验证实验确认敏感度排序可泛化至更高精度部署场景，且精度代价可忽略。实际收益为 ADC 面积削减——从 OPT-125M 的数十 mm² 扩展至 OPT-1.3B 的数百 mm²。

**局限性。** 线性替代模型在中等预算（10-30% 节省）下保持正确的分配排名，但在激进预算下可能退化，因为多组交互变得显著。组级分析是一阶信号；更精细的深度感知细化可进一步提升分配质量。更好的权重量化（如 GPTQ、AWQ）会降低 FP32 到 CIM 的绝对差距，但不会使 ADC 分配比较失效，因为该比较基于量化底线之上的相对敏感度差异。所有实验使用 NeuroSIM 仿真；硅验证及扩展至编码器-解码器或 MoE 架构为未来工作。

---

## 参考文献

| 编号 | 引用 |
|------|------|
| [1] | P. Chi et al., "PRIME: A novel processing-in-memory architecture for neural network computation in ReRAM-based main memory," *ISCA*, pp. 27-39, 2016. |
| [2] | A. Shafiee et al., "ISAAC: A convolutional neural network accelerator with in-situ analog arithmetic in crossbars," *ISCA*, pp. 14-26, 2016. |
| [3] | P.-Y. Chen, X. Peng, and S. Yu, "NeuroSim: A circuit-level macro model for benchmarking neuro-inspired architectures in online learning," *IEEE TCAD*, vol. 37, no. 12, pp. 3067-3080, 2018. |
| [4] | S. Zhang et al., "OPT: Open pre-trained transformer language models," *arXiv:2205.01068*, 2022. |
| [5] | Z. Dong et al., "HAWQ: Hessian aware quantization of neural networks with mixed precision," *ICCV*, pp. 293-302, 2019. |
| [6] | K. Wang et al., "HAQ: Hardware-aware automated quantization with mixed precision," *CVPR*, pp. 8612-8620, 2019. |
| [7] | T. Dettmers et al., "LLM.int8(): 8-bit matrix multiplication for transformers at scale," *NeurIPS*, vol. 35, pp. 30318-30332, 2022. |
| [8] | G. Xiao et al., "SmoothQuant: Accurate and efficient post-training quantization for large language models," *ICML*, pp. 38087-38099, 2023. |
| [9] | C. Wolters et al., "Memory is all you need: An overview of compute-in-memory architectures for accelerating large language model inference," *arXiv:2406.08413*, 2024. |
| [10] | J. Guo et al., "Accelerating LLM inference with flexible N:M sparsity via a fully digital compute-in-memory accelerator," *arXiv:2504.14365*, 2025. |
| [11] | J. Lim et al., "P3-LLM: An integrated NPU-PIM accelerator for LLM inference using hybrid numerical formats," *arXiv:2511.06838*, 2025. |
| [12] | A. Azamat et al., "Partial sum quantization for reducing ADC size in ReRAM-based neural network accelerators," *IEEE TCAD*, vol. 42, no. 12, pp. 4897-4908, 2023. |
| [13] | S. Sun et al., "CIM²PQ: An arraywise and hardware-friendly mixed precision quantization method for analog computing-in-memory," *IEEE TCAD*, vol. 43, no. 7, pp. 2084-2097, 2024. |
| [14] | G.-C. Chen et al., "Sensitivity-aware mixed-precision quantization for ReRAM-based computing-in-memory," *arXiv:2512.19445*, 2025. |
| [15] | W. Wan et al., "A mixed-precision memristor and SRAM compute-in-memory AI processor," *Nature*, vol. 639, no. 8055, pp. 617-623, 2025. |
| [16] | K. Danouchi et al., "MXFormer: A microscaling floating-point charge-trap transistor compute-in-memory transformer accelerator," *arXiv:2602.12480*, 2026. |
| [17] | E. Frantar et al., "GPTQ: Accurate post-training quantization for generative pre-trained transformers," *ICLR*, 2023. |
| [18] | J. Lin et al., "AWQ: Activation-aware weight quantization for on-device LLM compression and acceleration," *MLSys*, 2024. |
| [19] | Z. Yan et al., "Improving realistic worst-case performance of NVCiM DNN accelerators through training with right-censored Gaussian noise," *ICCAD*, pp. 1-9, 2023. |
| [20] | X. Peng et al., "DNN+NeuroSim V2.0: An end-to-end benchmarking framework for compute-in-memory accelerators for on-chip training," *IEEE TCAD*, vol. 40, no. 11, pp. 2306-2319, 2021. |
| [21] | J. Lee et al., "NeuroSim V1.4: Extending technology support for digital compute-in-memory toward 1nm node," *IEEE TCAS-I*, vol. 71, no. 4, pp. 1733-1744, 2024. |
| [22] | S. Merity et al., "Pointer sentinel mixture models," *ICLR*, 2017. |

---

## 附录：图表索引

| 编号 | 文件名 | 内容说明 |
|------|--------|----------|
| 图1 | fig_methodology | 提出的分配流程全景图 |
| 图2 | fig_overview_cim | LLM-to-CIM 映射与 ADC 面积动因 |
| 图3 | fig_ppa_curve | 芯片面积 vs. ADC 精度 & ADC 面积占比 |
| 图4 | fig2_saturation_heterogeneity | 逐层 ADC 饱和率（OPT-125M & 1.3B） |
| 图5 | fig_noise_path | 噪声传播假说示意图 |
| 图6 | fig_grouping_validity | 组级分析有效性：逐深度敏感度 |
| 图7 | fig_interaction_scatter | 替代模型可加性验证散点图 |
| 图8 | fig_deployment_regime | 跨场景基线稳定性 & 迁移分配 |
| 图9 | fig_sensitivity_heatmap | 实测敏感度热图 & 代理失效对比 |
| 图10 | fig_transfer_heatmap | 排名稳定性 & Spearman ρ 热图 |
| 图11 | fig5_bit_assignment | 逐层位宽分配可视化 |
| 图12 | fig6_pareto_frontier | PPL vs. ADC 面积 Pareto 前沿 |
| 图13 | fig_proxy_scatter | 代理指标失效散点图 |
| 图14 | fig_downstream_ci | 下游零样本任务精度 + 置信区间 |
| 图15 | fig_robustness_original | 噪声鲁棒性：增强噪声下敏感度柱状图 |

| 编号 | 内容说明 |
|------|----------|
| 表1 | 五维鲁棒性总结 |
| 表2 | 替代模型验证（分配最优性 + 逐对交互） |
| 表3 | 主实验：多信号多预算分配结果（OPT-125M） |
| 表4 | 跨模型验证 + OPT-1.3B 分配结果 |
| 算法1 | 组级 ADC 敏感度测量协议 |
| 公式1 | ADC 面积缩放模型 |
| 公式2 | Max-clip 饱和率定义 |
| 公式3 | 实测敏感度定义 |
| 公式4 | ILP 优化公式 |
| 公式5 | 混合精度 ADC 面积节省 |
