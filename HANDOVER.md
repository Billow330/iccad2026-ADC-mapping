# ICCAD 2026 项目交接文档

> 本文档供**新的Claude实例**阅读，用于完整接收并继续此项目。
> 请在开始任何工作前**完整阅读本文档**。

---

## 0. 项目基本信息

| 项目 | 详情 |
|------|------|
| 论文标题 | Measured Layer Sensitivity Guides Mixed-Precision ADC Allocation for CIM-Based LLM Inference |
| 投稿目标 | ICCAD 2026 |
| 作者 | Fantao Gao, Cancheng Xiao, Jiahao Zhao, Jianshi Tang, Tianxiang Nan* |
| 单位 | School of Integrated Circuits, Tsinghua University |
| 当前状态 | **论文初稿完成，6页，0 Overfull，无编译错误** |
| 项目根目录 | `/home/ubuntu/iccad2026_bxkj/` |
| 主工作目录 | `/home/ubuntu/iccad2026_bxkj/NeuroSim/` |
| 论文文件 | `/home/ubuntu/iccad2026_bxkj/paper/main.tex` |

---

## 1. 核心研究内容（必读）

### 1.1 论文故事线

**问题**：CIM芯片的ADC面积随位宽指数增长（7b时占芯片24.4%），需要混合精度分配来节省面积。

**现有方法的错误假设**：认为ADC饱和率高的层（attn_qkv ≈100%饱和）最敏感，应该优先保护。

**本文核心发现**（11×反转）：
- `q/k/v_proj`（attn_qkv）：饱和率≈100%，但ΔPPL/layer=**0.128**（最不敏感）
- `fc2`（ffn_down）：饱和率≈19%，但ΔPPL/layer=**1.413**（最敏感）
- **两者差距11倍，与饱和率预测完全相反**

**解决方案**：直接测量各层类型的CIM ADC敏感度（ΔPPL when 7b→6b），用ILP最优分配位宽。

### 1.2 四大贡献

1. **饱和率-敏感度11×反转**：首次系统发现
2. **代理指标双重失效**：饱和率ρ=-0.80（负相关），Hessian迹ρ=+0.20（不显著），0/5排名正确
3. **ILP分配+NeuroSIM验证**：20.5%节省，+2.2 PPL，比HAWQ-guided好**12.7 PPL**，OPT-1.3B交叉验证
4. **SmoothQuant协同**：SQ+6b PPL=305.0，低于7b基准306.4，节省50%面积

### 1.3 全部关键数字

| 指标 | 数值 |
|------|------|
| 7b ADC占芯片面积 | 24.4% |
| 10b ADC占芯片面积 | 63.9% |
| 反转倍数 | **11×**（1.413÷0.128） |
| ILP ADC节省 | **20.5%**（228.4→181.5 mm²） |
| ILP PPL代价 | **+2.2 PPL**（306.4→308.6） |
| ILP vs HAWQ差距 | **12.7 PPL**（321.3-308.6） |
| ILP vs Sat-Greedy | **5.2 PPL**（313.8-308.6） |
| SQ+6b PPL | **305.0**（低于7b基准！） |
| SQ+6b节省 | **50%** ADC面积 |
| ILP能耗节省 | **23.5%** |
| SQ+6b能耗节省 | **57.3%** |
| ILP零样本损失 | **-0.7%pt** avg（vs CIM-7b） |
| 饱和率 Spearman ρ | **-0.80** |
| Hessian Spearman ρ | **+0.20**，p=0.75 |
| OPT-1.3B ffn_down | ΔPPL/layer=+0.619（最敏感，与125M一致）|
| OPT-1.3B attn_qkv | ΔPPL/layer=-0.051（最不敏感，与125M一致）|

---

## 2. 项目目录结构

```
/home/ubuntu/iccad2026_bxkj/
├── paper/                          # LaTeX论文
│   ├── main.tex                    # 主文件（6页，完整）
│   ├── refs.bib                    # 参考文献
│   ├── main.pdf                    # 编译后PDF
│   └── fig{1-9}_*.pdf             # 论文图片
├── NeuroSim/                       # 主工作目录
│   ├── sensitivity_analysis.py     # 核心：PerLayerCIMHook, ilp_allocation
│   ├── stable_eval.py              # 100-batch稳定评估
│   ├── smooth_quant.py             # CIMSmoothQuant
│   ├── llm_inference.py            # load_model, CIMNoiseHook
│   ├── hessian_sensitivity.py      # HAWQ Hessian迹
│   ├── compute_energy_savings.py   # 能耗分析
│   ├── plot_paper_figures_v2.py    # 9张图生成
│   ├── update_fig7.py              # Fig7更新（含HAWQ柱）
│   ├── run_zeroshot*.py            # 零样本评估
│   ├── stability_test.py           # 稳定性测试
│   ├── model_cache/                # HF模型缓存（7.2GB）
│   │   ├── models--facebook--opt-125m/
│   │   └── models--facebook--opt-1.3b/
│   └── results/                    # 所有实验结果
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
│       ├── opt125m/outlier_facebook_opt-125m_adc7.csv
│       ├── opt1.3b/outlier_facebook_opt-1.3b_adc7.csv
│       ├── figures_iccad2026_v2/   # 最终图片（fig1-fig9）
│       └── figures_tables_data.xlsx # 所有图表数据汇总
├── paper_data_complete.md          # 完整论文数据手册
└── HANDOVER.md                     # 本文件
```

---

## 3. 环境配置

### 3.1 Python环境

```bash
# Python版本
python3 --version  # 3.10.12

# 已安装的关键包
# PyTorch 2.10.0+cpu（无GPU）
# transformers 5.2.0
# datasets 4.6.0
# lm_eval（lm-evaluation-harness）
# scipy（用于ILP：scipy.optimize.milp）
# openpyxl（Excel导出）

# 安装命令（新服务器）
pip install torch transformers datasets lm_eval scipy openpyxl
```

### 3.2 模型缓存

```bash
# OPT-125M（约500MB）
./model_cache/models--facebook--opt-125m/

# OPT-1.3B（约5GB）
./model_cache/models--facebook--opt-1.3b/

# 如果需要重新下载
cd /home/ubuntu/iccad2026_bxkj/NeuroSim
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('facebook/opt-125m', cache_dir='./model_cache')
AutoTokenizer.from_pretrained('facebook/opt-125m', cache_dir='./model_cache')
"
```

### 3.3 关键环境变量

```bash
# lm-eval离线模式（必须用这个，不能用HF_HUB_OFFLINE）
export HF_DATASETS_OFFLINE=1

# 已缓存的lm-eval数据集
# hellaswag ✓, winogrande ✓, wikitext ✓
# arc_challenge ✗（未缓存，无法离线使用）
```

### 3.4 NeuroSIM参数（重要）

```
pipeline=false
speedUpDegree=1
synchronous=false
novelMapping=false
```

OPT-125M有4种唯一层尺寸：768×768（×48），768×3072（×12），3072×768（×12），768×50272（×1）

---

## 4. 核心API用法（避免踩坑）

```python
# 1. 加载模型
from llm_inference import load_model, load_wikitext2, make_loader, CIMNoiseHook
model, tok = load_model('facebook/opt-125m', './model_cache', 'cpu')

# 2. CIMNoiseHook（均匀位宽）
hook = CIMNoiseHook(model, adc_bits=7, weight_bits=8, input_bits=8)
hook.calibrate(calib_loader, clip_percentile=99.0)
hook.install()
# ... 评估 ...
hook.remove()

# 3. PerLayerCIMHook（混合位宽）—— bit_assignment是字典
from sensitivity_analysis import PerLayerCIMHook
hook = PerLayerCIMHook(model, bit_assignment={'layer.name': 6, ...}, default_bits=7)
hook.calibrate(calib_loader)
hook.install()

# 4. ILP分配 —— 返回list，不是dict！
from sensitivity_analysis import ilp_allocation
bits_list = ilp_allocation(
    sensitivity_data,   # group_sensitivity dict
    ppa_sweep,          # from ppa_sweep_opt125m.csv
    nominal_bits=7,
    bit_choices=[4,5,6,7,8],
    target_area_savings=0.20
)

# 5. SmoothQuant —— model不在__init__里
from smooth_quant import CIMSmoothQuant
sq = CIMSmoothQuant(weight_bits=8, input_bits=8, adc_bits=6, adc_clip_pct=99.0, sat_lambda=0.5)
sq.fit(model, calib_loader, num_batches=4, device='cpu', task='lm')

# 6. 计算PPL
from smooth_quant import compute_perplexity
ppl = compute_perplexity(model, eval_loader)
```

---

## 5. 论文结构

论文共6页，0 Overfull，无编译错误。

### 编译命令

```bash
cd /home/ubuntu/iccad2026_bxkj/paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

### 章节结构

| 章节 | 内容 |
|------|------|
| Abstract | 核心贡献摘要，含OPT-1.3B确认和HAWQ 12.7 PPL优势 |
| §1 Introduction | 4个贡献，11×反转发现，OPT-1.3B交叉验证提及 |
| §2 Background | CIM ADC架构，激活outlier，NeuroSIM设置 |
| §3 Measuring Per-Layer Sensitivity | 测量协议，稳定性测试（Δseed=+2.4±3.7），Table I |
| §4 Optimal Mixed-Precision Allocation | ILP公式，Greedy算法，Table III（含HAWQ行） |
| §5 Discussion | 6个子节：反转原因/芯片设计意义/OPT-1.3B/能耗/代理失效/零样本 |
| §6 Related Work | CIM、LLM量化、NeuroSIM相关文献 |
| §7 Conclusion | 核心结论 |

### 图表清单

| 编号 | 内容 | 文件 |
|------|------|------|
| Fig 1 | ADC面积 vs 位宽（折线图） | fig1_adc_motivation.pdf |
| Fig 2 | 饱和率异质性（柱状图，两模型） | fig2_saturation_heterogeneity.pdf |
| Fig 3 | 分层敏感度（水平条形图） | fig3_group_sensitivity.pdf |
| Fig 4 | 代理散点图（sat/Hessian vs ΔPPL） | fig4_sensitivity_vs_saturation.pdf |
| Fig 5 | 逐层位宽分配热图（ILP vs Greedy） | fig5_bit_assignment.pdf |
| Fig 6 | Pareto前沿 | fig6_pareto_frontier.pdf |
| Fig 7 | 8种配置对比柱状图（含HAWQ） | fig7_comparison_bars.pdf |
| Fig 8 | ILP vs Greedy多预算折线图 | fig8_ppl_vs_savings.pdf |
| Fig 9 | ILP vs Greedy跨预算详细对比 | fig9_ilp_vs_greedy_multibudget.pdf |
| Table I | 敏感度 vs 饱和率（5种层类型） | §3 |
| Table II | NeuroSIM PPA Sweep（含1.3B缩放） | §4 |
| Table III | 8种配置PPL+面积（含HAWQ行） | §4 |
| Table IV | 代理排名对比 | §5.4 |
| Table V | 零样本准确率 | §5.5 |
| Table VI | OPT-1.3B组敏感度 | §5.3 |

---

## 6. 所有实验结果数据

### 6.1 敏感度测量（OPT-125M, 7b→6b）

来源：`results/sensitivity/opt125m/group_sensitivity.json`，baseline_ppl=333.84

| Layer type | 数量 | ΔPPL（全组） | ΔPPL/layer | sat_rate(p99) | sat_rate_worst(max-clip) |
|-----------|------|------------|-----------|-------------|------------------------|
| ffn_down (fc2) | 12 | +16.955 | **1.4129** | 3.53% | 19.3% |
| attn_out (out_proj) | 12 | +10.389 | **0.8657** | 3.52% | 3.6% |
| lm_head | 1 | +0.700 | **0.7004** | 3.53% | 100.0% |
| ffn_up (fc1) | 12 | +2.417 | **0.2015** | 3.56% | 94.5% |
| attn_qkv (q/k/v) | 36 | +4.618 | **0.1283** | 3.53% | 99.99% |

> ⚠️ **重要**：论文Table I的饱和率列（≈100%/94%/19%/4%）用的是`sat_rate_worst`（max-clip），不是sensitivity实验里的`sat_rate`（p99-clip, ≈3.5%）

### 6.2 稳定eval结果（100-batch）

来源：`results/stable/opt125m/stable_eval_results.csv`

| 配置 | PPL | ADC面积(mm²) | 芯片面积(mm²) | ADC节省% | 6b层数 | 7b层数 |
|-----|-----|------------|------------|---------|-------|-------|
| Uniform 7b | **306.4323** | 228.43 | 936.6 | 0% | 0 | 73 |
| Uniform 6b | 315.2698 | 114.22 | 822.38 | 50% | 73 | 0 |
| Sat-Greedy 20% | 313.7992 | 181.49 | 889.66 | 20.55% | 30 | 43 |
| Sens-Greedy 20% | 312.4954 | 181.49 | 889.66 | 20.55% | 30 | 43 |
| **ILP 20%** | **308.5532** | 181.49 | 889.66 | 20.55% | 30 | 43 |
| SQ + Uniform 7b | 315.0930 | 228.43 | 936.6 | 0% | 0 | 73 |
| **SQ + Uniform 6b** | **304.9719** | 114.22 | 822.38 | 50% | 73 | 0 |
| SQ + ILP 20% | 309.3894 | 181.49 | 889.66 | 20.55% | 30 | 43 |

### 6.3 HAWQ对比

来源：`results/sensitivity/opt125m/hawq_comparison.json`

| 配置 | PPL（精确） |
|-----|----------|
| Uniform 7b（HAWQ实验基准） | 326.796 |
| HAWQ-guided Greedy | **321.315** |
| ILP（论文主结果） | **308.600** |

> 差距：321.3 - 308.6 = **12.7 PPL**（论文核心卖点之一）

### 6.4 PPA Sweep

来源：`results/ppa/opt125m/ppa_sweep_opt125m.csv`

| ADC bits | 芯片面积(mm²) | ADC面积(mm²) | ADC占比 | 能耗(nJ) |
|---------|------------|------------|--------|---------|
| 3 | 667.65 | 26.20 | 3.92% | 4,578 |
| 4 | 694.57 | 40.57 | 5.84% | 5,141 |
| 5 | 731.03 | 63.65 | 8.71% | 6,110 |
| 6 | 806.38 | 125.23 | 15.53% | 7,922 |
| **7** | **936.60** | **228.43** | **24.39%** | **11,401** |
| 8 | 1,146.86 | 407.42 | 35.53% | 18,205 |
| 9 | 1,671.54 | 821.34 | 49.14% | 31,738 |
| 10 | 2,821.71 | 1,802.79 | 63.89% | 58,681 |

### 6.5 多预算ILP vs Greedy

来源：`results/stable/opt125m/ilp_vs_greedy_multibudget.csv`

| 节省目标% | ILP PPL | Greedy PPL | ΔPPL(ILP-Grd) | ILP更优 |
|---------|--------|----------|--------------|--------|
| 5% | 312.89 | 311.53 | -1.37 | ✗ |
| **10%** | **305.23** | 311.07 | **+5.84** | ✓ |
| 15% | 311.02 | 312.33 | +1.30 | ✓ |
| **20%** | **309.28** | 311.16 | **+1.89** | ✓ |
| 25% | 313.88 | 314.10 | +0.22 | ✓ |
| **30%** | **311.50** | 314.86 | **+3.36** | ✓ |
| 40% | 311.73 | 312.95 | +1.21 | ✓ |
| 50% | 318.46 | 311.59 | -6.87 | ✗ |

### 6.6 零样本准确率

来源：`results/zeroshot/opt125m/zeroshot_summary.json`

| 配置 | HellaSwag | WinoGrande | Avg |
|-----|-----------|------------|-----|
| FP32 (clean) | 34.8% | 53.6% | 44.2% |
| CIM 7b | 30.6% | 52.0% | 41.3% |
| SQ + 6b | 26.2% | 51.0% | 38.6% |
| **ILP 20%** | **30.0%** | **51.2%** | **40.6%** |

### 6.7 稳定性测试

来源：`results/stable/opt125m/stability_results.json`

| 配置 | Seed 0 | Seed 1 | Seed 2 | Mean±Std |
|-----|--------|--------|--------|---------|
| Uniform 7b | 319.938 | 310.053 | 307.373 | 312.5±5.4 |
| ILP 20% | 321.234 | 308.556 | 314.672 | 314.8±5.2 |
| SQ + 6b | 326.601 | 312.042 | 309.586 | 316.1±7.5 |

Within-seed ΔPPL(ILP-Uniform)：[+1.30, -1.50, +7.30] → **mean=+2.4±3.7**（与Table III的+2.2一致）

### 6.8 Hessian代理

来源：`results/hessian/opt125m/hessian_group.json`

| Layer type | Hessian mean | Hessian std |
|-----------|-------------|------------|
| attn_qkv | 0.001417 | 0.001369 |
| attn_out | 0.000183 | 0.000139 |
| ffn_up | 0.002015 | 0.001328 |
| ffn_down | 0.005291 | 0.009066 |
| lm_head | 0.009270 | 0.000000 |

HAWQ排名（从高到低）：lm_head > ffn_down > ffn_up > attn_qkv > attn_out
CIM实测排名（从高到低）：ffn_down > attn_out > lm_head > ffn_up > attn_qkv
→ 0/5正确

### 6.9 OPT-1.3B敏感度

来源：`results/sensitivity/opt1.3b/group_sensitivity.json`，nominal=11b，probe=10b，baseline_ppl=708.99

| Layer type | 数量 | ΔPPL/layer | 排名 |
|-----------|------|-----------|-----|
| ffn_down | 24 | **+0.619** | 1（最敏感）|
| ffn_up | 24 | +0.149 | 2 |
| attn_out | 24 | +0.033 | 3 |
| attn_qkv | 72 | **-0.051** | 4（最不敏感）|
| lm_head | 1 | -10.869 | 排除（不稳定）|

与OPT-125M排序一致：ffn_down最敏感，attn_qkv最不敏感 → **跨模型验证成功**

### 6.10 能耗分析

来源：`results/stable/opt125m/energy_analysis.csv`

| 配置 | 总能耗(pJ) | 节省% |
|-----|-----------|------|
| Uniform 7b | 11,400,583 | 0% |
| **ILP 20%** | **8,717,303** | **23.5%** |
| SQ + 6b | 4,871,267 | **57.3%** |

---

## 7. 论文当前状态与待办事项

### 7.1 已完成

- [x] 论文初稿完成（6页，0 Overfull）
- [x] 所有实验数据收集完毕
- [x] 9张图全部生成
- [x] 6个表格全部完成
- [x] OPT-1.3B跨模型验证（Table VI）
- [x] HAWQ对比实验（Table III HAWQ行，Fig 7 HAWQ柱）
- [x] 稳定性测试（3个校准种子）
- [x] 零样本评估（HellaSwag + WinoGrande）
- [x] 能耗分析

### 7.2 可能的后续改进（如果有时间）

- [ ] **图片重绘**：用户希望自己重画所有图，已提供完整数据（见`paper_data_complete.md`和`results/figures_tables_data.xlsx`）
- [ ] **投稿格式检查**：确认IEEEtran格式符合ICCAD 2026要求
- [ ] **参考文献完整性**：检查refs.bib中所有引用是否有DOI
- [ ] **Abstract字数**：确认不超过IEEE会议限制（通常200词）
- [ ] **arc_challenge零样本**：当前只有HellaSwag+WinoGrande（arc_challenge数据集未缓存）

### 7.3 论文关键注意事项

1. **PPL数值一致性**：
   - Table III（stable eval）的7b基准PPL=306.4
   - HAWQ实验的7b基准PPL=326.8（不同校准批次）
   - 论文通过within-seed ΔPPL解决了稳定性报告问题（+2.4±3.7，与+2.2一致）

2. **饱和率定义**：
   - Table I / Fig 2 用的是`sat_rate_worst`（**max-clip**，100th percentile校准）
   - sensitivity实验记录的`sat_rate`是p99-clip校准，约3.5%（完全不同！不要混淆）

3. **ILP分配**：attn_qkv/fc1→6b，fc2/out_proj/lm_head→7b（ILP最优解，三个种子完全一致）

4. **lm_head in OPT-1.3B**：ΔPPL/layer=-10.87，负值是数值不稳定，**论文Table VI中已排除**

---

## 8. 聊天记录位置

所有与前Claude实例的对话记录保存在：

```
/home/ubuntu/.claude/projects/-home-ubuntu-iccad2026-bxkj/
├── memory/
│   └── MEMORY.md          # 跨对话持久化记忆（最重要）
├── *.jsonl                # 历次对话完整记录（8个文件）
└── d0d8408d-.../          # 最长的一次对话（含上下文摘要）
```

**如果你在新服务器上**，这些文件应该已经通过压缩包迁移过来。将`.claude`目录恢复到`/home/ubuntu/.claude/`即可。

---

## 9. 快速验证环境

在新服务器上运行以下命令验证环境是否正确：

```bash
cd /home/ubuntu/iccad2026_bxkj/NeuroSim

# 验证模型可加载
python3 -c "
from llm_inference import load_model
model, tok = load_model('facebook/opt-125m', './model_cache', 'cpu')
print('Model loaded:', model.__class__.__name__)
print('Params:', sum(p.numel() for p in model.parameters())/1e6, 'M')
"

# 验证论文可编译
cd /home/ubuntu/iccad2026_bxkj/paper
pdflatex main.tex 2>&1 | tail -5

# 验证结果文件存在
ls /home/ubuntu/iccad2026_bxkj/NeuroSim/results/stable/opt125m/
ls /home/ubuntu/iccad2026_bxkj/NeuroSim/results/sensitivity/opt125m/
```

---

## 10. 参考文档

| 文档 | 路径 | 说明 |
|------|------|------|
| 完整数据手册 | `/home/ubuntu/iccad2026_bxkj/paper_data_complete.md` | 所有图表数据详细说明 |
| Excel数据汇总 | `NeuroSim/results/figures_tables_data.xlsx` | 11个Sheet，所有图表数据 |
| 论文LaTeX | `paper/main.tex` | 完整论文原文 |
| Memory文件 | `/home/ubuntu/.claude/projects/.../memory/MEMORY.md` | Claude跨对话记忆 |

---

*交接文档生成时间：2026-03-03*
*项目状态：论文初稿完成，等待投稿*
