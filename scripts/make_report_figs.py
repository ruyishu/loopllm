"""生成 loop 技术报告的图表（复刻参考图风格：浅灰底、hatch 纹理、数值标签、Figure 标题）。
输出到 report_figs/*.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.facecolor": "#f7f8f9",
    "axes.edgecolor": "#2b2b2b",
    "axes.linewidth": 1.1,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "grid.color": "#c9ccd1",
    "grid.linestyle": ":",
    "grid.linewidth": 0.9,
    "figure.facecolor": "white",
    "legend.frameon": False,
    "font.size": 11,
})

BLUE, LBLUE, GRAY, TAN = "#3f7fd0", "#8fc0ee", "#c6d0da", "#d9d2c2"
HATCH = "///"


def cap(ax_fig, n, text, y=0.015):
    """图下注释：Figure N | ... （Latn 用衬线体，中文回退到雅黑，避免缺字形）"""
    ax_fig.text(0.5, y, "Figure %d | %s" % (n, text), ha="center", va="bottom",
                fontsize=11.5, family=["Times New Roman", "Microsoft YaHei"], color="#1a1a1a")


def style_ax(ax, ymax=None, ylab=None):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color("#2b2b2b")
    if ymax:
        ax.set_ylim(0, ymax)
    if ylab:
        ax.set_ylabel(ylab, fontsize=12)
    ax.tick_params(colors="#2b2b2b")


def grouped_bar(n, name, groups, series, ylab, caption, ymax=None, pct=True, figsize=(11.4, 6.2), note=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.085, 0.20, 0.895, 0.66])
    k = len(series)
    x = np.arange(len(groups))
    w = 0.78 / k
    for i, (label, vals, color, hatch) in enumerate(series):
        xs = x - 0.39 + w * (i + 0.5)
        draw = [0 if v is None else v for v in vals]
        bars = ax.bar(xs, draw, width=w * 0.92, label=label, color=color,
                      edgecolor="#5b6b7c", linewidth=0.7, hatch=hatch, zorder=3)
        for b, v in zip(bars, vals):
            if v is None:          # 尚未产出的权重：留空，避免被读成 0 分
                continue
            ax.text(b.get_x() + b.get_width() / 2, v + (1.6 if pct else 0.03),
                    ("%.1f" % v) if pct else ("%.3f" % v), ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold", color="#111111", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11.5)
    if pct:
        ax.set_ylim(0, ymax or max(max([v for v in s[1] if v is not None]) for s in series) * 1.25)
    style_ax(ax, ymax=None if pct else None, ylab=ylab)
    if ymax:
        ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(MultipleLocator(20 if pct else 0.1))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=k, fontsize=11.5)
    if note:
        fig.text(0.085, 0.115, note, fontsize=10, color="#555555", ha="left")
    cap(fig, n, caption)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


# ---------------- 数据（全部来自实测日志，见报告附录） ----------------
STEPS = [1000, 5000, 10000, 20000, 30000, 39695]
L_T1 = [2.1109, 1.9992, 1.8531, 1.8470, 1.7276, 1.6496]
L_T4 = [2.0785, 1.9525, 1.8193, 1.7942, 1.6910, 1.5853]

# 表 1/图 1：20 题数学 ToolUse 通过率（math_only 口径）
W = ["pretrain", "full_sft", "dpo", "grpo", "agent"]
T1_MINI = [0, 30, None, 30, 30]
T4_MINI = [0, 50, 50, 50, 50]
T1_FULL = [0, 35, 35, 35, None]

# 图 4：RL Reward（每 ~90 / 60 步一个点）
GR_ST = [1, 91, 181, 271, 361, 451, 541, 631, 721, 811, 901, 991, 1081, 1171, 1261, 1351, 1441, 1531, 1621, 1711, 1801,
         1891, 1981, 2071, 2161, 2251, 2341, 2431, 2521, 2611, 2701, 2791, 2881, 2971, 3061, 3151, 3241, 3331, 3421,
         3511, 3601, 3691, 3781, 3871, 3961, 4051, 4141, 4231, 4321, 4411, 4501, 4591, 4681, 4771, 4861, 4951, 5041,
         5131, 5221, 5311, 5401, 5491, 5581, 5671, 5761, 5851, 5941, 6031, 6121, 6211, 6301, 6391, 6481, 6571, 6661,
         6751, 6841, 6931, 7021, 7111, 7201, 7291, 7381, 7471, 7561, 7651, 7741, 7831, 7921, 8011, 8101, 8191, 8281,
         8371, 8461, 8551, 8641, 8731, 8821, 8911, 9001, 9091, 9181, 9271, 9361, 9451, 9541, 9631, 9721]
GR_RW = [-1.255, -0.609, -0.157, -1.328, -0.516, 0.971, -1.661, -0.534, 0.887, 0.342, 0.124, -1.398, -0.585, 1.142,
         -1.342, -1.376, -0.279, -1.384, -0.382, -0.992, -0.909, -1.634, -0.675, 0.164, -1.288, 0.933, -0.117, -1.425,
         1.156, 0.278, 1.217, 0.849, 1.259, -0.15, -1.743, 1.727, 0.307, 2.124, 1.173, -2.223, -1.074, -0.725, -3.39,
         -0.237, 0.898, 0.734, -0.679, -1.594, 0.748, 2.011, 1.89, 1.228, 0.977, -1.594, 0.764, -0.082, -0.957, 3.155,
         0.737, -1.362, 0.065, 1.093, 0.668, -0.733, 0.229, 1.435, -0.322, 0.091, 2.167, 1.343, 0.765, 0.923, 1.089,
         0.825, 1.691, 0.455, -1.251, 0.058, -0.585, 1.376, 0.591, -0.376, -1.853, 0.188, 2.564, 1.498, 0.683, 1.552,
         0.644, -0.548, 1.04, -2.144, 0.081, -0.297, 3.761, 0.675, -0.326, 1.988, 1.492, 1.125, 0.158, -0.134, -2.033,
         0.764, -1.399, 1.226, 1.644, 1.892, -0.013]
AG_ST = [1, 61, 121, 181, 241, 301, 361, 421, 481, 541, 601, 661, 721, 781, 841, 901, 961, 1021, 1081, 1141, 1201, 1261,
         1321, 1381, 1441, 1501, 1561, 1621, 1681, 1741, 1801, 1861, 1921, 1981, 2041, 2101, 2161, 2221, 2281, 2341,
         2401, 2461, 2521, 2581, 2641, 2701, 2761, 2821, 2881, 2941, 3001, 3061, 3121, 3181, 3241, 3301, 3361, 3421,
         3481, 3541, 3601, 3661, 3721, 3781, 3841, 3901, 3961, 4021, 4081, 4141, 4201]
AG_RW = [-1.391, -0.212, -1.584, -0.519, -0.646, -0.632, -0.248, -0.594, -1.434, 2.987, -0.726, 0.579, 0.155, -1.617,
         -0.43, 0.253, -0.631, 0.139, -2.021, 0.405, -0.674, 1.432, 1.345, -1.157, -1.411, -1.528, -1.337, -0.3, 0.819,
         -0.641, 0.783, 0.406, 0.596, -0.759, -0.31, 0.152, -1.133, -0.728, 1.313, -0.502, 0.818, 1.59, 0.949, -0.317,
         -0.412, -0.821, 0.705, 1.138, 1.807, 1.217, -0.174, -0.818, -0.757, -0.574, 0.273, -0.803, -1.398, -1.133,
         0.183, 2.042, -0.487, -1.611, -0.436, 0.207, -1.236, -1.584, -1.069, -0.129, -0.722, -1.558, 0.616]

# ============ Figure 1 ============
grouped_bar(1, "fig1_downstream.png", W,
            [("T=1 · mini 1.7G", [v if v is not None else 0 for v in T1_MINI], BLUE, HATCH),
             ("T=4 · mini 1.7G（+loop）", [v if v is not None else 0 for v in T4_MINI], LBLUE, HATCH),
             ("T=1 · 完整 14G", [v if v is not None else 0 for v in T1_FULL], TAN, HATCH)],
            "20 题数学 ToolUse 通过率 (%)", "同深度加循环（T=4）的下游收益：SFT 之后全线 +20 个百分点",
            ymax=65, note="口径：20 道固定题 + 标准答案自动校验，工具集只给 calculate_math；dpo/agent 行的空缺表示该权重尚未产出（进行中）。")

# ============ Figure 2：loss 曲线 ============
fig = plt.figure(figsize=(11.4, 6.2))
ax = fig.add_axes([0.085, 0.20, 0.895, 0.66])
ax.plot(STEPS, L_T1, "o-", color=BLUE, lw=2.4, ms=7, label="T=1（原深度）", zorder=3)
ax.plot(STEPS, L_T4, "s-", color="#e8a33d", lw=2.4, ms=7, label="T=4（+loop）", zorder=3)
for s, a, b in zip(STEPS, L_T1, L_T4):
    ax.annotate("%.4f" % a, (s, a), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9.5, color="#2b4c7e")
    ax.annotate("%.4f" % b, (s, b), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=9.5, color="#a06a12")
ax.set_xlabel("训练步数（mini 预训练，各 39,695 步）")
ax.set_ylabel("loss")
ax.set_ylim(1.52, 2.18)
style_ax(ax, ylab="loss")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.19), ncol=2, fontsize=11.5)
cap(fig, 2, "预训练 loss 对照：同数据、同超参、同参数量（63.91M），唯一差别是循环轮数 T", 0.045)
fig.text(0.085, 0.105, "终态 loss：T=1 为 1.6496，T=4 为 1.5853（低 0.064，相对 −3.9%）。",
         fontsize=10, color="#555555")
fig.savefig(os.path.join(OUT, "fig2_loss.png"), dpi=200)
plt.close(fig)

# ============ Figure 3：同步数 loss 差值 ============
fig = plt.figure(figsize=(11.4, 5.6))
ax = fig.add_axes([0.085, 0.22, 0.895, 0.62])
diffs = [b - a for a, b in zip(L_T1, L_T4)]
bars = ax.bar([str(s) for s in STEPS], diffs, width=0.52, color="#e8a33d", edgecolor="#b07a15",
              linewidth=0.8, hatch=HATCH, zorder=3)
for b, v in zip(bars, diffs):
    ax.text(b.get_x() + b.get_width() / 2, v - 0.006, "%.3f" % v, ha="center", va="top",
            fontsize=11, fontweight="bold", color="#7a4d00", zorder=4)
ax.axhline(0, color="#2b2b2b", lw=1.0)
ax.set_xlabel("训练步数（同 token 数对齐）")
ax.set_ylim(-0.09, 0.02)
style_ax(ax, ylab="Δ loss（T=4 − T=1）")
cap(fig, 3, "同步数损失差：全程为负，说明加循环后每一步都比原深度更好", 0.04)
fig.savefig(os.path.join(OUT, "fig3_diff.png"), dpi=200)
plt.close(fig)

# ============ Figure 4：RL Reward ============
def roll(v, k=7):
    out = []
    for i in range(len(v)):
        lo = max(0, i - k + 1)
        out.append(float(np.mean(v[lo:i + 1])))
    return out

fig = plt.figure(figsize=(11.4, 6.4))
ax = fig.add_axes([0.085, 0.20, 0.895, 0.64])
ax.plot(GR_ST, GR_RW, color="#b9d4f2", lw=1.1, alpha=0.9, zorder=2, label="RLAIF_FULL 原始（批均值）")
ax.plot(GR_ST, roll(GR_RW), color=BLUE, lw=2.6, zorder=4, label="RLAIF_FULL 滑动均值")
ax.plot(AG_ST, AG_RW, color="#f0d8b4", lw=1.1, alpha=0.9, zorder=2, label="AGENT_FULL 原始（批均值）")
ax.plot(AG_ST, roll(AG_RW), color="#e8a33d", lw=2.6, zorder=4, label="AGENT_FULL 滑动均值")
ax.axhline(0, color="#777777", lw=1.0, ls="--", zorder=1)
ax.annotate("前 500 步均值 −0.944", (500, -0.944), textcoords="offset points", xytext=(-8, -26),
            fontsize=10.5, color="#2b4c7e", ha="left")
ax.annotate("后 500 步均值 +0.587", (9200, 0.587), textcoords="offset points", xytext=(-70, 18),
            fontsize=10.5, color="#2b4c7e", ha="left")
ax.set_xlabel("训练步数")
ax.set_ylabel("Reward（批均值）")
ax.set_ylim(-3.9, 4.4)
style_ax(ax, ylab="Reward")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.19), ncol=2, fontsize=10.5)
cap(fig, 4, "完整档 RL 段的 Reward 轨迹：窗口均值 −0.944 → +0.587（RLAIF 跑满整轮，agent 段仍在进行）", 0.045)
fig.text(0.085, 0.10, "判据：Reward 上行 + KL_ref 受控（−0.03~−0.15）→ 真在学习，而非 reward hacking。",
         fontsize=10, color="#555555")
fig.savefig(os.path.join(OUT, "fig4_rl.png"), dpi=200)
plt.close(fig)

# ============ Figure 5：工具集大小的影响 ============
grouped_bar(5, "fig5_toolset.png", ["8 个工具", "3 个工具", "1 个工具"],
            [("调用了计算器的比例", [0, 85, 100], GRAY, HATCH),
             ("答案正确率", [0, 15, 30], BLUE, HATCH)],
            "比例 (%)", "工具集大小是 64M 模型的硬瓶颈：给 8 个工具时完全不调工具",
            ymax=120, figsize=(10.2, 5.8),
            note="同一批题目、同一个 base-mini full_sft 权重；8 个工具时连仓库自带的 demo 题（256×37）也不再调用工具。")

# ============ Figure 6：两条完整档链路时间线 ============
fig = plt.figure(figsize=(12.6, 5.8))
ax = fig.add_axes([0.175, 0.24, 0.80, 0.60])
# (label, 起, 止, 颜色, 是否已完成)   —— 时间用"9/15 以来的小时"表达（h(日, 时, 分)）
def h(day, hour, minute=0):
    return (day - 15) * 24 + hour + minute / 60.0
rows = [
    ("mini 对照实验（9/15）", h(15, 10, 38), h(15, 23, 38), "#b9c6d3", True),
    ("base(T=1) 预训练", h(15, 23, 2), h(16, 15, 53), GRAY, True),
    ("base SFT 完整档", h(16, 15, 54), h(17, 14, 19), GRAY, True),
    ("base DPO", h(17, 14, 19), h(17, 14, 29), GRAY, True),
    ("base RLAIF_FULL", h(17, 14, 19), h(18, 0, 57), BLUE, True),
    ("base AGENT_FULL", h(18, 0, 57), h(19, 19, 15), LBLUE, False),
    ("T=4 预训练", h(17, 1, 10), h(19, 15, 10), "#e8a33d", False),
    ("T=4 SFT（预计）", h(19, 15, 10), h(22, 20, 0), TAN, False),
    ("T=4 RL（预计）", h(22, 20, 0), h(24, 24, 0), TAN, False),
]
for i, (label, s, e, c, done) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.barh(y, e - s, left=s, height=0.56, color=c, edgecolor="#5b6b7c", linewidth=0.7,
            hatch=None if done else "\\\\\\", zorder=3)
    ax.text(e + 1.5, y, ("已完成" if done else "进行中/预计"), va="center", fontsize=9.5,
            color="#3d7a3d" if done else "#a06a12")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=11)
ticks = [h(d, 0) for d in range(16, 26)]
ax.set_xticks(ticks)
ax.set_xticklabels(["9/%d" % d for d in range(16, 26)], fontsize=10.5)
ax.set_xlim(h(15, 12), h(25, 12))
ax.axvline(h(18, 9, 50), color="#cc3333", lw=1.4, ls="--", zorder=4)
ax.text(h(18, 9, 50), len(rows) - 0.35, " 现在", color="#cc3333", fontsize=10.5, va="top")
style_ax(ax)
ax.xaxis.grid(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
cap(fig, 6, "两条完整档复现链路的时间线（实测 + 预计）：base 走完 RL 全轮，T=4 因算力需 ~9/25 收尾", 0.035)
fig.savefig(os.path.join(OUT, "fig6_timeline.png"), dpi=200)
plt.close(fig)

# ============ Figure 7：代价对照 ============
grouped_bar(7, "fig7_cost.png", ["每步耗时（相对）", "吞吐 tok/s", "等效 MFU (%)"],
            [("T=1（原深度）", [1.0, 99.3, 32.0], BLUE, HATCH),
             ("T=4（+loop）", [3.8, 28.1, 36.0], "#e8a33d", HATCH)],
            "数值（不同量纲，仅示意相对关系）",
            "代价：T=4 每步约 3.5~3.9× 时间；按等效算力折算的 MFU 与原深度相当（实现无明显浪费）",
            ymax=120, figsize=(10.6, 5.8),
            note="实测（完整档 SFT 段，seq 768 / batch 16）：T=1 为 8.08 步/秒、T=4 为 2.29 步/秒；tok/s 取训练日志稳态值。")

print("FIGS_OK")
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    print(f, os.path.getsize(p) // 1024, "KB")
