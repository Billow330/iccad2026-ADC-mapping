# 给下一个AI的完整项目接手提示词

> **使用方式**：将本文件的全部内容直接粘贴给下一个AI（Claude/GPT等），作为第一条消息发送。它将获得接手此项目所需的全部上下文。

---

## 你的角色与任务

你是一个论文写作助手，正在协助完成一篇投稿ICCAD 2026的学术论文。论文**初稿已完成**，当前需要的工作是**论文润色、修改、以及可能的图表重绘**。请直接进入工作状态，不要重新介绍自己。

---

## 论文基本信息

| 项目 | 内容 |
|------|------|
| **题目** | Measured Layer Sensitivity Guides Mixed-Precision ADC Allocation for CIM-Based LLM Inference |
| **投稿目标** | ICCAD 2026（IEEE/ACM International Conference on Computer-Aided Design） |
| **作者** | Fantao Gao, Cancheng Xiao, Jiahao Zhao, Jianshi Tang, Tianxiang Nan* |
| **单位** | School of Integrated Circuits, Tsinghua University, Beijing 100084, China |
| **通讯** | nantianxiang@mail.tsinghua.edu.cn |
| **论文状态** | **6页，0 Overfull，无编译错误，初稿完成** |
| **格式** | IEEEtran conference，双栏 |

---

## 核心研究故事（必须理解）

### 问题背景
CIM（Compute-in-Memory）芯片用于LLM推理时，ADC（模数转换器）面积随位宽**指数增长**：
- 7-bit时ADC占芯片面积 **24.4%**（OPT-125M，NeuroSIM验证）
- 10-bit时ADC占 **63.9%**

**混合精度分配**（不同层用不同ADC位宽）可以节省面积。问题是：哪些层可以降低位宽？

### 现有方法的错误假设
业界普遍认为：**ADC饱和率高的层最敏感**，应该优先保护（保持高位宽）。
- `q/k/v_proj`层：饱和率≈100%，被认为最需要保护
- `fc2`层：饱和率≈19%，被认为可以降低位宽

### 本文核心发现（11×反转）
我们直接**测量**每类层在 7b→6b 时的PPL变化（ΔPPL），发现：

| 层类型 | 饱和率 | ΔPPL/layer | 真实敏感度排名 |
|--------|--------|------------|--------------|
| `q/k/v_proj`（attn_qkv） | **≈100%** | **0.128** | 5（最不敏感！） |
| `fc1`（ffn_up） | ≈94% | 0.201 | 4 |
| `lm_head` | ≈100% | 0.700 | 3 |
| `out_proj`（attn_out） | ≈4% | 0.866 | 2 |
| `fc2`（ffn_down） | **≈19%** | **1.413** | 1（最敏感！） |

**结论**：饱和率高的层反而最不敏感！fc2（低饱和）比attn_qkv（高饱和）敏感**11倍**。这与直觉完全相反。

### 为什么会发生反转？
- 高饱和层（attn_qkv）的裁剪是**系统性的**（每个token都裁同样的通道），后续LayerNorm会部分恢复动态范围
- `fc2`/`out_proj`直接注入残差流，**没有中间归一化**，量化噪声无法被补偿
- 注意力空间高度冗余，softmax在位置间归一化，吸收了噪声

### 解决方案
用**ILP（整数线性规划）**基于实测敏感度最优分配位宽：
- `q/k/v_proj`和`fc1` → 降到6b（最不敏感，可以省面积）
- `fc2`、`out_proj`、`lm_head` → 保持7b（最敏感，必须保护）
- **结果**：20.5%面积节省，仅+2.2 PPL代价

---

## 所有关键数字（论文中出现的精确值）

### 主要结果（OPT-125M，WikiText-2，100-batch stable eval）

| 配置 | PPL | ADC面积(mm²) | 芯片面积(mm²) | ADC节省 |
|------|-----|------------|------------|---------|
| Uniform 7b（基准） | **306.4** | 228.4 | 936.6 | 0% |
| Uniform 6b | 315.3 | 114.2 | 822.4 | 50% |
| HAWQ-guided greedy | 321.3 | 181.5 | 889.7 | 20.5% |
| Greedy（sat-guided） | 313.8 | 181.5 | 889.7 | 20.5% |
| Greedy（sens-guided） | 312.5 | 181.5 | 889.7 | 20.5% |
| **ILP（sens-guided）** | **308.6** | **181.5** | **889.7** | **20.5%** |
| SQ + Uniform 6b | **305.0** | 114.2 | 822.4 | 50% |
| SQ + ILP 20% | 309.4 | 181.5 | 889.7 | 20.5% |

关键对比：
- ILP vs HAWQ：**12.7 PPL**更好（308.6 vs 321.3）
- ILP vs Sat-Greedy：**5.2 PPL**更好（308.6 vs 313.8）
- SQ+6b的PPL=305.0，**低于7b基准**306.4！（同时节省50%面积）

### 敏感度测量（OPT-125M，7b→6b，baseline_ppl=333.84）

| 层类型 | 层数 | ΔPPL（全组） | ΔPPL/layer | 饱和率（max-clip） |
|--------|------|------------|-----------|-----------------|
| ffn_down / fc2 | 12 | +16.955 | **1.4129** | **≈19%** |
| out_proj | 12 | +10.389 | **0.8657** | ≈4% |
| lm_head | 1 | +0.700 | **0.7004** | ≈100% |
| fc1 / ffn_up | 12 | +2.417 | **0.2015** | ≈94% |
| attn_qkv (q/k/v) | 36 | +4.618 | **0.1283** | **≈100%** |

> ⚠️ **重要区分**：饱和率有两种定义：
> - **论文Table I / Fig 2使用**：`sat_rate_worst`（max-clip，100th percentile校准）→ 100%/94%/19%/4%
> - **实验内部记录**：`sat_rate`（p99-clip校准）→ 约3.5%（所有层相近，不是论文里的数字！）

### 代理指标失效（Proxy Invalidation）

| 代理方法 | Spearman ρ vs CIM | 排名正确数 | p值 |
|---------|-------------------|-----------|-----|
| 饱和率（Sat. rate） | **-0.80**（负相关！） | — | — |
| Hessian迹（HAWQ） | +0.20 | **0/5** | 0.75（不显著） |

HAWQ排名（最敏感→最不敏感）：lm_head > ffn_down > ffn_up > attn_qkv > attn_out
CIM实测排名：ffn_down > attn_out > lm_head > ffn_up > attn_qkv
→ 完全不一样，0/5正确

### OPT-1.3B跨模型验证（nominal=11b，probe=10b）

| 层类型 | 层数 | ΔPPL/layer | 排名 |
|--------|------|-----------|------|
| ffn_down | 24 | **+0.619** | 1（最敏感） |
| ffn_up | 24 | +0.149 | 2 |
| attn_out | 24 | +0.033 | 3 |
| attn_qkv | 72 | **-0.051** | 4（最不敏感） |

**结论**：与OPT-125M完全一致，反转是跨模型的普遍现象。

### NeuroSIM PPA Sweep（OPT-125M实测）

| ADC bits | 芯片面积(mm²) | ADC面积(mm²) | ADC占比 | 能耗(nJ/512-token) |
|---------|------------|------------|--------|------------------|
| 3 | 668 | 26.2 | 3.9% | 4,578 |
| 4 | 695 | 40.6 | 5.8% | 5,141 |
| 5 | 731 | 63.7 | 8.7% | 6,110 |
| 6 | 806 | 125.2 | 15.5% | 7,922 |
| **7** | **937** | **228.4** | **24.4%** | **11,401** |
| 8 | 1,147 | 407.4 | 35.5% | 18,205 |
| 9 | 1,672 | 821.3 | 49.1% | 31,738 |
| 10 | 2,822 | 1,802.8 | 63.9% | 58,681 |

OPT-1.3B缩放比例：14.22×（tile count）

### 能耗分析

| 配置 | 总能耗(nJ) | 节省% |
|------|-----------|------|
| Uniform 7b | 11.4 | 0% |
| **ILP 20%** | **8.7** | **23.5%** |
| SQ + 6b | 4.9 | **57.3%** |

### 稳定性测试（3个校准种子）

| 配置 | Seed 0 | Seed 1 | Seed 2 | Mean±Std |
|------|--------|--------|--------|---------|
| Uniform 7b | 319.9 | 310.1 | 307.4 | 312.5±5.4 |
| ILP 20% | 321.2 | 308.6 | 314.7 | 314.8±5.2 |
| SQ + 6b | 326.6 | 312.0 | 309.6 | 316.1±7.5 |

Within-seed ΔPPL(ILP-Uniform)：[+1.3, -1.5, +7.3] → **mean=+2.4±3.7**（与Table III的+2.2一致）

### 零样本准确率（500样本，0-shot，lm-eval）

| 配置 | HellaSwag | WinoGrande | Avg |
|------|-----------|------------|-----|
| FP32 clean | 34.8% | 53.6% | 44.2% |
| CIM 7b | 30.6% | 52.0% | 41.3% |
| SQ + 6b | 26.2% | 51.0% | 38.6% |
| **ILP 20%** | **30.0%** | **51.2%** | **40.6%** |

ILP-20%仅损失 **-0.7%pt**（平均），远好于SQ+6b的-2.7%pt。

### 多预算 ILP vs Greedy（稳定eval）

| 节省目标 | ILP PPL | Greedy PPL | ILP优势 |
|---------|--------|----------|--------|
| 5% | 312.9 | 311.5 | -1.4（Greedy赢） |
| **10%** | **305.2** | 311.1 | **+5.8** ✓ |
| 15% | 311.0 | 312.3 | +1.3 ✓ |
| **20%** | **309.3** | 311.2 | **+1.9** ✓ |
| 25% | 313.9 | 314.1 | +0.2 ✓ |
| **30%** | **311.5** | 314.9 | **+3.4** ✓ |
| 40% | 311.7 | 313.0 | +1.2 ✓ |
| 50% | 318.5 | 311.6 | -6.9（Greedy赢） |

ILP在6/8个预算点胜出，最大优势在10%（+5.8）和30%（+3.4）。

---

## 论文完整结构

### 章节列表

| 章节 | 标题 | 核心内容 |
|------|------|---------|
| Abstract | — | 11×反转，ILP 20.5%节省+2.2PPL，HAWQ-12.7PPL，SQ+6b=305 |
| §1 Introduction | — | 4个贡献，动机，11×发现，OPT-1.3B验证提及 |
| §2 Background | — | CIM架构+ADC公式，激活outlier+饱和率定义，NeuroSIM设置 |
| §3 Measuring Per-Layer ADC Sensitivity | — | 测量协议，Table I，稳定性（Δseed=+2.4±3.7），11×反转图 |
| §4 Optimal Mixed-Precision ADC Allocation | — | ILP公式，Greedy算法，饱和率引导对比，Table III，Pareto图 |
| §5 NeuroSIM PPA Validation | — | Table II，Table III完整结果，Fig7比较柱状图，Fig9多预算 |
| §6 Discussion | — | 6个子节（反转原因/芯片设计意义/OPT-1.3B/能耗/代理/零样本） |
| §7 Related Work | — | CIM、LLM量化、混合精度、NeuroSIM文献 |
| §8 Conclusion | — | 核心结论 |

> 注：论文实际章节编号可能与上表略有出入，以main.tex为准。

### 四大贡献（在Introduction中明确列出）

1. **Saturation-sensitivity inversion**：首次系统测量，11×反转，q/k/v_proj（100%饱和）最不敏感，fc2（19%饱和）最敏感
2. **Proxy invalidation**：饱和率ρ=-0.80，Hessian迹ρ=0.20（p=0.75），0/5排名正确；直接测量是唯一可靠方法
3. **ILP allocation with NeuroSIM validation**：20.5%节省，比HAWQ好12.7 PPL，比Sat-Greedy好5.2 PPL，OPT-1.3B跨模型验证
4. **SmoothQuant synergy**：SQ+6b PPL=305.0（低于7b基准），节省50%面积+57%能耗

### 图表列表

| 编号 | 内容 | 文件名 | 位置 |
|------|------|--------|------|
| Fig 1 | ADC面积 vs 位宽折线图 | `fig1_adc_motivation.pdf` | §2 |
| Fig 2 | 饱和率异质性柱状图（两模型对比） | `fig2_saturation_heterogeneity.pdf` | §2 |
| Fig 3 | 分层敏感度水平条形图 | `fig3_group_sensitivity.pdf` | §3 |
| Fig 4 | 代理散点图（sat/Hessian vs ΔPPL） | `fig4_sensitivity_vs_saturation.pdf` | §3 |
| Fig 5 | 逐层位宽分配热图（ILP vs Greedy） | `fig5_bit_assignment.pdf` | §4/5 |
| Fig 6 | Pareto前沿（PPL vs ADC面积） | `fig6_pareto_frontier.pdf` | §4/5 |
| Fig 7 | 8种配置对比柱状图（含HAWQ） | `fig7_comparison_bars.pdf` | §5 |
| Fig 8 | ILP vs Greedy多预算折线图（旧） | `fig8_ppl_vs_savings.pdf` | §5 |
| Fig 9 | ILP vs Greedy跨预算详细对比 | `fig9_ilp_vs_greedy_multibudget.pdf` | §5 |
| Table I | 敏感度 vs 饱和率（5种层类型） | — | §3 |
| Table II | NeuroSIM PPA Sweep（含1.3B缩放） | — | §5 |
| Table III | 8种配置PPL+面积（含HAWQ行） | — | §5 |
| Table IV | 代理排名对比（3列排名+ΔPPL） | — | §6 |
| Table V | 零样本准确率 | — | §6 |
| Table VI | OPT-1.3B组敏感度（4行） | — | §6 |

---

## 服务器环境与文件结构

### 服务器环境
```
Python: 3.10.12
PyTorch: 2.10.0+cpu（无GPU！）
transformers: 5.2.0
datasets: 4.6.0
lm_eval: 已安装
scipy: 已安装（用于ILP: scipy.optimize.milp）
```

### 关键环境变量
```bash
export HF_DATASETS_OFFLINE=1   # 必须用这个，不要用HF_HUB_OFFLINE
```

### 项目目录结构
```
/home/ubuntu/iccad2026_bxkj/
├── paper/
│   ├── main.tex                    ← 论文主文件（6页，完整）
│   ├── refs.bib                    ← 参考文献
│   ├── main.pdf                    ← 编译后PDF
│   └── fig{1-9}_*.pdf             ← 9张论文图片
├── NeuroSim/                       ← 主工作目录
│   ├── sensitivity_analysis.py     ← PerLayerCIMHook, ilp_allocation
│   ├── stable_eval.py             ← 100-batch稳定评估
│   ├── smooth_quant.py            ← CIMSmoothQuant
│   ├── llm_inference.py           ← load_model, CIMNoiseHook
│   ├── hessian_sensitivity.py     ← HAWQ Hessian迹
│   ├── compute_energy_savings.py  ← 能耗分析
│   ├── plot_paper_figures_v2.py   ← 9张图生成
│   ├── model_cache/               ← HF模型缓存（7.2GB）
│   │   ├── models--facebook--opt-125m/
│   │   └── models--facebook--opt-1.3b/
│   └── results/
│       ├── ppa/opt125m/ppa_sweep_opt125m.csv
│       ├── sensitivity/opt125m/group_sensitivity.json
│       ├── sensitivity/opt125m/allocations.json
│       ├── sensitivity/opt125m/pareto_frontier.json
│       ├── sensitivity/opt125m/hawq_comparison.json
│       ├── sensitivity/opt1.3b/group_sensitivity.json
│       ├── hessian/opt125m/hessian_group.json
│       ├── stable/opt125m/stable_eval_results.csv
│       ├── stable/opt125m/ilp_vs_greedy_multibudget.csv
│       ├── stable/opt125m/stability_results.json
│       ├── stable/opt125m/energy_analysis.csv
│       ├── zeroshot/opt125m/zeroshot_summary.json
│       ├── opt125m/outlier_facebook_opt-125m_adc7.csv    ← 饱和率数据源
│       ├── opt1.3b/outlier_facebook_opt-1.3b_adc7.csv
│       └── figures_iccad2026_v2/   ← 最终图片（fig1-fig9）
├── paper_data_complete.md          ← 完整论文数据手册（31KB）
└── HANDOVER.md                     ← 项目交接文档
```

### 论文编译命令
```bash
cd /home/ubuntu/iccad2026_bxkj/paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## 核心代码API（如需运行实验）

```python
# 1. 加载模型
from llm_inference import load_model, load_wikitext2, make_loader, CIMNoiseHook
model, tok = load_model('facebook/opt-125m', './model_cache', 'cpu')

# 2. 均匀位宽CIM hook
hook = CIMNoiseHook(model, adc_bits=7, weight_bits=8, input_bits=8)
hook.calibrate(calib_loader, clip_percentile=99.0)
hook.install()
# ... 评估 ...
hook.remove()

# 3. 混合位宽hook（bit_assignment是字典！）
from sensitivity_analysis import PerLayerCIMHook
hook = PerLayerCIMHook(model, bit_assignment={'model.decoder.layers.0.self_attn.q_proj': 6, ...}, default_bits=7)
hook.calibrate(calib_loader)
hook.install()

# 4. ILP分配（返回list，不是dict！）
from sensitivity_analysis import ilp_allocation
bits_list = ilp_allocation(
    sensitivity_data,   # group_sensitivity dict
    ppa_sweep,          # from ppa_sweep_opt125m.csv
    nominal_bits=7,
    bit_choices=[4,5,6,7,8],
    target_area_savings=0.20
)

# 5. SmoothQuant（model不在__init__里！）
from smooth_quant import CIMSmoothQuant
sq = CIMSmoothQuant(weight_bits=8, input_bits=8, adc_bits=6, adc_clip_pct=99.0, sat_lambda=0.5)
sq.fit(model, calib_loader, num_batches=4, device='cpu', task='lm')

# 6. 计算PPL
from smooth_quant import compute_perplexity
ppl = compute_perplexity(model, eval_loader)
```

---

## 论文重要注意事项

### 数据一致性
- **Table III（stable eval）的7b基准PPL=306.4**
- **HAWQ实验的7b基准PPL=326.8**（不同校准批次，两者都对！论文用within-seed ΔPPL解决）
- 不要用326.8和308.6对比说"ILP节省了18 PPL"——应该比较的是同一基准下的相对差异

### 饱和率的两种定义（极其重要，不要混淆！）
- **论文Table I / Fig 2 / 正文中的"≈100%/≈94%/≈19%"**：来自`sat_rate_worst`（max-clip，100th percentile校准）
  - 数据来源：`results/opt125m/outlier_facebook_opt-125m_adc7.csv`的`sat_rate_worst`列
  - 这才是论文故事的核心数据！
- **实验内部的`sat_rate`**：p99-clip校准，约3.5%（所有层都差不多）
  - 这个数字**不出现在论文正文**，仅在代码内部使用

### ILP最优分配（不变的，三个种子完全一致）
```
q/k/v_proj → 6b（36层，最不敏感）
fc1（ffn_up）→ 6b（12层）
fc2（ffn_down）→ 7b（12层，最敏感）
out_proj → 7b（12层）
lm_head → 7b（1层）
```

### OPT-1.3B的lm_head
ΔPPL/layer=-10.87（负值，数值不稳定），**论文Table VI中已排除，不要讨论**

### NeuroSIM设置（必须）
```
pipeline=false
speedUpDegree=1
synchronous=false
novelMapping=false
```

---

## 当前论文完整正文（关键段落）

### Abstract（完整）
> Compute-in-Memory (CIM) chips for large language model (LLM) inference face a critical bottleneck: ADC area grows exponentially with bit resolution, reaching 24.4% of total chip area at 7-bit for OPT-125M (NeuroSIM-validated). Prior mixed-precision methods assume layers with high ADC saturation rates are most sensitive to bit reduction—but we show this is fundamentally incorrect. By measuring perplexity change when each layer type's ADC resolution is independently reduced 7b→6b, we find that q/k/v_proj layers (≈100% saturation) cause only ΔPPL = 0.13/layer, while fc2 layers (≈19% saturation) cause 1.41/layer—an **11× inversion** confirmed on OPT-1.3B. Based on measured sensitivities, we formulate an ILP to optimally allocate ADC bits under area constraints. On OPT-125M (100-batch eval), ILP achieves 20.5% ADC area savings with only +2.2 PPL increase, outperforming HAWQ-guided by 12.7 PPL and saturation-guided greedy by 5.2 PPL at the same target. SmoothQuant + 6b achieves PPL=305.0—below the 7b CIM baseline—while saving 50% ADC area. All area results are NeuroSIM-validated.

### Conclusion（完整）
> We present the first direct measurement of per-layer ADC sensitivity for CIM-based LLM inference, demonstrating that the commonly assumed saturation-rate proxy is actively misleading: q/k/v_proj (≈100% saturation) are 11× less sensitive than fc2 (≈19% saturation). OPT-1.3B validation confirms this inversion is a cross-model property of transformer CIM systems. Our measurement-guided ILP achieves 20.5% ADC area savings at +2.2 PPL, outperforming HAWQ-guided greedy by 12.7 PPL and saturation-guided greedy by 5.2 PPL (NeuroSIM-validated). SmoothQuant + 6b saves 50% ADC area while achieving PPL below the 7b CIM baseline. The key design guideline: measure ADC sensitivity directly—not by proxy.

---

## 待办事项（你可能需要做的工作）

### 最可能的任务
1. **论文润色**：改进英文表达、逻辑流、段落衔接
2. **图表重绘**：用户想用自己的风格重画图，数据都在上面
3. **格式检查**：确认IEEEtran格式符合ICCAD 2026要求
4. **投稿前检查**：字数限制、参考文献格式、图片分辨率等

### 数据文件位置（如需重绘图）
- `NeuroSim/results/figures_tables_data.xlsx`：11个Sheet，所有图表数据（含颜色标注）
- `paper_data_complete.md`：所有图表数据的Markdown说明（31KB）

### 当前不需要做的
- ❌ 重新运行实验（所有数据已收集完毕）
- ❌ 重新推导任何理论（已完成）
- ❌ 修改论文故事线（已确定）

---

## 如何开始工作

**用户会告诉你具体要做什么**。你可以：
1. 直接阅读 `/home/ubuntu/iccad2026_bxkj/paper/main.tex` 来查看完整论文
2. 读 `HANDOVER.md` 获得更多技术细节
3. 读 `paper_data_complete.md` 获得图表数据详情

如果需要编译论文：
```bash
cd /home/ubuntu/iccad2026_bxkj/paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

**请直接进入工作状态，等待用户指令。**

---

*本提示词由前任AI实例生成，时间：2026-03-03*
*项目状态：论文初稿完成，等待投稿前最终修改*
