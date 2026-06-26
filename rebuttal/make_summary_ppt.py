#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a one-slide .pptx summarizing the ICCAD #707 rebuttal work (Chinese)."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

FONT = "Microsoft YaHei"          # CJK font; viewer substitutes if absent
NAVY = RGBColor(0x1F, 0x38, 0x64)
HEAD = RGBColor(0x2E, 0x5B, 0x9A)
LIGHT = RGBColor(0xEC, 0xF1, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY = RGBColor(0x6B, 0x6B, 0x6B)
DARK = RGBColor(0x22, 0x22, 0x22)
ACCENT = RGBColor(0xC0, 0x39, 0x2B)


def cjk(run, name=FONT):
    run.font.name = name
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {}); rPr.append(el)
        el.set("typeface", name)


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # ---- title ----
    tb = slide.shapes.add_textbox(Inches(0.4), Inches(0.22), Inches(12.5), Inches(0.9))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = "ICCAD 2026 #707  Rebuttal 工作总结"
    r.font.size = Pt(26); r.font.bold = True; r.font.color.rgb = NAVY; cjk(r)
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = "Measured Layer Sensitivity Guides Mixed-Precision ADC Allocation for CIM-Based LLM Inference"
    r2.font.size = Pt(11); r2.font.italic = True; r2.font.color.rgb = GRAY; cjk(r2)

    # ---- table ----
    headers = ["审稿人核心质疑", "补充实验", "关键结果"]
    rows = [
        ("机理只是定性假设", "707A-W1 · 707C-W2/C3/Q2 · 707D-W3",
         "解析分解 + 诊断", "Table S4",
         "D_gran ×4.06 / D_ovl ×1.00;局部误差跨度 740× → 输出漂移仅 1.6×(解耦)⇒ 必须直接测 ΔPPL"),
        ("baseline 不可用(PPL 36→300)", "707C-C7 · 707D-W1/C1/Q1",
         "OPT-125M 可用工作点", "max-clip",
         "权重 INT8 几乎无损(42.5→42.5);max-clip 8b → PPL 93;Wfc2 最敏感、FFN ≫ attention"),
        ("缺现代模型完整流程", "707A-W2 · 707C-C8",
         "Qwen2-7B 完整 profiling→ILP→PPA", "Table S3",
         "省 34% ADC 面积+能耗 @ +1.8% PPL;下游 PIQA/BoolQ 差 ~1.6pp;proxy-blind → PPL 1938"),
        ("PPL 作优化信号是否稳健", "707C-C2/Q1",
         "token 级 NLL 重算", "Table S1",
         "逐组排序 Spearman ρ = 1.0,ILP 分配完全不变(PPL=exp(NLL) 单调,可证)"),
        ("排序鲁棒性", "707A-W1 · 707C-Q2 · 707D-W3",
         "探针深度 × 种子扫描", "Table S2",
         "FFN > attention 在 7/9 设置成立;7→4 探针下 ≈ 11×"),
        ("ADC 实现 / 硬件开销", "707B-W1 · 707D-W2/Q2/Q3",
         "实现细节 + 开销澄清", "CR-4",
         "45nm / MLSA / 128×128 RRAM;group 级无额外 mux/路由/时序,仅 per-array 位宽寄存器"),
    ]
    nrows = len(rows) + 1
    left, top, width, height = Inches(0.4), Inches(1.35), Inches(12.55), Inches(5.35)
    gt = slide.shapes.add_table(nrows, 3, left, top, width, height)
    tbl = gt.table
    tbl.columns[0].width = Inches(3.15)
    tbl.columns[1].width = Inches(2.75)
    tbl.columns[2].width = Inches(6.65)
    # disable default banding styling by setting fills manually
    tbl.first_row = True

    def put(cell, lines, size, bold, color, align=PP_ALIGN.LEFT, sub=None):
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = Inches(0.08); cell.margin_right = Inches(0.08)
        cell.margin_top = Inches(0.03); cell.margin_bottom = Inches(0.03)
        tf = cell.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.alignment = align
        r = p.add_run(); r.text = lines
        r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; cjk(r)
        if sub:
            p2 = tf.add_paragraph(); p2.alignment = align
            r2 = p2.add_run(); r2.text = sub
            r2.font.size = Pt(8.5); r2.font.color.rgb = GRAY; cjk(r2)

    # header row
    for j, h in enumerate(headers):
        c = tbl.cell(0, j)
        c.fill.solid(); c.fill.fore_color.rgb = HEAD
        put(c, h, 13, True, WHITE, PP_ALIGN.CENTER)
    tbl.rows[0].height = Inches(0.45)

    # body rows
    for i, (q, qtag, ex, extag, res) in enumerate(rows, start=1):
        shade = WHITE if i % 2 else LIGHT
        c0 = tbl.cell(i, 0); c0.fill.solid(); c0.fill.fore_color.rgb = shade
        put(c0, q, 11.5, True, DARK, sub=qtag)
        c1 = tbl.cell(i, 1); c1.fill.solid(); c1.fill.fore_color.rgb = shade
        put(c1, ex, 11, True, HEAD, sub=extag)
        c2 = tbl.cell(i, 2); c2.fill.solid(); c2.fill.fore_color.rgb = shade
        put(c2, res, 10.5, False, DARK)
        tbl.rows[i].height = Inches((5.35 - 0.45) / len(rows))

    # ---- footer conclusion ----
    fb = slide.shapes.add_textbox(Inches(0.4), Inches(6.78), Inches(12.55), Inches(0.6))
    ftf = fb.text_frame; ftf.word_wrap = True
    fp = ftf.paragraphs[0]; fp.alignment = PP_ALIGN.LEFT
    fr = fp.add_run()
    fr.text = ("结论:三类核心质疑(机理定性 · baseline 不可用 · 缺现代模型与硬件细节)"
               "全部转化为新证据 —— 逐条有据、与论文自洽、诚实可复现。")
    fr.font.size = Pt(12); fr.font.bold = True; fr.font.color.rgb = ACCENT; cjk(fr)

    out = "/home/scratch.gr212_prime_h0_2/gr212/nvgpu_gr212/layout/revP3.0/pandr/partitions/billowg/iccad_v2/rebuttal/rebuttal_summary_1page.pptx"
    prs.save(out)
    print("saved:", out)


if __name__ == "__main__":
    main()
