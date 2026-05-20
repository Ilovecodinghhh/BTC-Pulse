"""
Generate presentation slides for BTC-Pulse project review.
Style: Simple, clean, logical. Dark theme with accent colors.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

# ── Font & Style Setup ─────────────────────────────────────
from matplotlib.font_manager import FontProperties
FONT_CN = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
FONT_CN_BOLD = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
FONT_CN_LIGHT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Light.ttc')

# Colors
BG = '#0F1419'
BG_CARD = '#1A2332'
BG_CARD2 = '#1E2D3D'
WHITE = '#E8EAED'
GRAY = '#8899AA'
ACCENT = '#00D4AA'   # teal
ACCENT2 = '#4A9EFF'  # blue
ACCENT3 = '#FF6B6B'  # red
ACCENT4 = '#FFD93D'  # yellow
ACCENT5 = '#C084FC'  # purple
DIM = '#556677'

OUT = Path(__file__).parent
W, H = 16, 9
DPI = 150


def new_slide():
    fig, ax = plt.subplots(figsize=(W, H))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    return fig, ax


def add_page_num(ax, num, total=12):
    ax.text(15.5, 0.3, f'{num}/{total}', fontproperties=FONT_CN_LIGHT,
            fontsize=10, color=DIM, ha='right', va='bottom')


def add_top_bar(ax, left_text='BTC-Pulse', right_text=''):
    ax.plot([0.5, 15.5], [8.5, 8.5], color=DIM, linewidth=0.5, alpha=0.3)
    ax.text(0.8, 8.7, left_text, fontproperties=FONT_CN_LIGHT,
            fontsize=9, color=DIM, va='bottom')
    if right_text:
        ax.text(15.2, 8.7, right_text, fontproperties=FONT_CN_LIGHT,
                fontsize=9, color=DIM, va='bottom', ha='right')


def draw_card(ax, x, y, w, h, color=BG_CARD, alpha=0.8):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='none', alpha=alpha)
    ax.add_patch(rect)


# ═══════════════════════════════════════════════════════════
# SLIDE 1: Title
# ═══════════════════════════════════════════════════════════
def slide_01_title():
    fig, ax = new_slide()

    # Decorative line
    ax.plot([3, 13], [5.5, 5.5], color=ACCENT, linewidth=2, alpha=0.6)

    ax.text(8, 7.0, 'BTC-Pulse', fontproperties=FONT_CN_BOLD,
            fontsize=48, color=WHITE, ha='center', va='center')
    ax.text(8, 5.9, '多维度比特币趋势预测系统',
            fontproperties=FONT_CN, fontsize=22, color=ACCENT, ha='center')

    ax.text(8, 4.6, '多Agent协作  ·  量化回测  ·  统计验证  ·  隐私合规',
            fontproperties=FONT_CN_LIGHT, fontsize=14, color=GRAY, ha='center')

    # Bottom info
    ax.text(8, 2.5, 'v2.0  |  MIT License  |  100% 免费数据源',
            fontproperties=FONT_CN_LIGHT, fontsize=12, color=DIM, ha='center')
    ax.text(8, 1.8, 'github.com/Ilovecodinghhh/BTC-Pulse',
            fontproperties=FONT_CN_LIGHT, fontsize=11, color=ACCENT2, ha='center')

    add_page_num(ax, 1)
    fig.savefig(OUT / 'slide_01_title.png', dpi=DPI, bbox_inches='tight',
                facecolor=BG, pad_inches=0.3)
    plt.close()


# ═══════════════════════════════════════════════════════════
# SLIDE 2: Problem Statement
# ═══════════════════════════════════════════════════════════
def slide_02_problem():
    fig, ax = new_slide()
    add_top_bar(ax, 'BTC-Pulse', '痛点分析')

    ax.text(8, 7.8, '为什么需要 BTC-Pulse？',
            fontproperties=FONT_CN_BOLD, fontsize=30, color=WHITE, ha='center')

    problems = [
        ('▼', '情绪驱动', '散户恐慌/贪婪时做出非理性决策，\n单一指标无法全面判断市场状态', ACCENT3),
        ('▼', '回测失真', '多数回测忽略交易成本、前瞻偏差，\n导致策略上线后表现远低于预期', ACCENT4),
        ('▼', '过拟合幻觉', '参数优化后Sharpe很高，\n但无法区分是真实alpha还是数据挖掘', ACCENT5),
    ]

    for i, (icon, title, desc, color) in enumerate(problems):
        x = 1.5 + i * 4.5
        draw_card(ax, x, 2.5, 4.0, 4.3, BG_CARD)
        ax.text(x + 2.0, 6.2, icon, fontsize=32, ha='center', va='center')
        ax.text(x + 2.0, 5.3, title, fontproperties=FONT_CN_BOLD,
                fontsize=18, color=color, ha='center')
        ax.text(x + 2.0, 4.2, desc, fontproperties=FONT_CN_LIGHT,
                fontsize=11, color=GRAY, ha='center', va='top', linespacing=1.6)

    ax.text(8, 1.5, '→  BTC-Pulse：多维数据融合 + 严谨统计验证 + 全链路闭环',
            fontproperties=FONT_CN, fontsize=14, color=ACCENT, ha='center')

    add_page_num(ax, 2)
    fig.savefig(OUT / 'slide_02_problem.png', dpi=DPI, bbox_inches='tight',
                facecolor=BG, pad_inches=0.3)
    plt.close()


# ═══════════════════════════════════════════════════════════
# SLIDE 3: Architecture
# ═══════════════════════════════════════════════════════════
def slide_03_architecture():
    fig, ax = new_slide()
    add_top_bar(ax, 'BTC-Pulse', '系统架构')

    ax.text(8, 8.0, '五层流水线架构',
            fontproperties=FONT_CN_BOLD, fontsize=28, color=WHITE, ha='center')

    layers = [
        ('数据采集', 'Bitcoinity · CCXT\nFNG · RSS', ACCENT2, 1.0),
        ('存储', 'SQLite\n增量更新', ACCENT, 4.0),
        ('特征工程', '50+指标\n多周期融合', ACCENT4, 7.0),
        ('模型信号', '规则60%+ML40%\nXGBoost·LLM', ACCENT5, 10.0),
        ('展示', 'Streamlit\n实时仪表盘', ACCENT3, 13.0),
    ]

    for name, desc, color, x in layers:
        draw_card(ax, x, 3.8, 2.5, 3.2, BG_CARD)
        ax.text(x + 1.25, 6.5, name, fontproperties=FONT_CN_BOLD,
                fontsize=14, color=color, ha='center')
        ax.text(x + 1.25, 5.2, desc, fontproperties=FONT_CN_LIGHT,
                fontsize=10, color=GRAY, ha='center', va='top', linespacing=1.5)

    # Arrows
    for i in range(4):
        x = 3.5 + i * 3.0
        ax.annotate('', xy=(x + 0.8, 5.4), xytext=(x, 5.4),
                    arrowprops=dict(arrowstyle='->', color=DIM, lw=1.5))

    # Bottom: Freqtrade bridge
    draw_card(ax, 3.5, 1.0, 9.0, 2.0, BG_CARD2)
    ax.text(8, 2.5, 'Freqtrade Bridge 专业回测层',
            fontproperties=FONT_CN_BOLD, fontsize=13, color=ACCENT, ha='center')
    ax.text(8, 1.7, 'Strategy · Backtester · RiskManager · Hyperopt · DataProvider',
            fontproperties=FONT_CN_LIGHT, fontsize=10, color=GRAY, ha='center')

    add_page_num(ax, 3)
    fig.savefig(OUT / 'slide_03_architecture.png', dpi=DPI, bbox_inches='tight',
                facecolor=BG, pad_inches=0.3)
    plt.close()


# ═══════════════════════════════════════════════════════════
# SLIDE 4: Innovation
# ═══════════════════════════════════════════════════════════
def slide_04_innovation():
    fig, ax = new_slide()
    add_top_bar(ax, 'BTC-Pulse', '创新性 25%')

    ax.text(8, 8.0, '创新性：三重验证闭环',
            fontproperties=FONT_CN_BOLD, fontsize=28, color=WHITE, ha='center')

    items = [
        ('1', '多维数据融合', '链上 + 情绪 + 衍生品 + 技术面 + 新闻NLP\n打破单一指标局限', ACCENT2),
        ('2', '规则+ML混合信号', '60%规则(可解释) + 40%XGBoost(泛化)\nML不可用时优雅降级', ACCENT),
        ('3', '统计严谨性验证', '缩水Sharpe + Bootstrap CI + 随机基准MC\n从"看起来赚钱"到"统计上显著"', ACCENT4),
        ('4', '开发流程创新', 'Coder+Reviewer双Agent对抗协作\n7轮迭代、评分≥9/10方可通过', ACCENT5),
    ]

    for i, (num, title, desc, color) in enumerate(items):
        y = 6.5 - i * 1.6
        # Number circle
        circle = plt.Circle((1.5, y), 0.35, color=color, alpha=0.2)
        ax.add_patch(circle)
        ax.text(1.5, y, num, fontproperties=FONT_CN_BOLD,
                fontsize=16, color=color, ha='center', va='center')
        # Text
        ax.text(2.5, y + 0.15, title, fontproperties=FONT_CN_BOLD,
                fontsize=15, color=WHITE, va='center')
        ax.text(2.5, y - 0.45, desc, fontproperties=FONT_CN_LIGHT,
                fontsize=10, color=GRAY, va='top', linespacing=1.5)

    # Right side: key differentiator box
    draw_card(ax, 10, 2.0, 5.5, 5.5, BG_CARD)
    ax.text(12.75, 7.0, '核心差异化', fontproperties=FONT_CN_BOLD,
            fontsize=14, color=ACCENT, ha='center')
    diff_text = (
        '传统回测：\n'
        '  收益率高 → "策略有效"\n\n'
        'BTC-Pulse：\n'
        '  收益率高\n'
        '  + DSR > 0.95  (非过拟合)\n'
        '  + p < 0.05    (胜过随机)\n'
        '  + CI不含零    (统计显著)\n'
        '  → "策略可能有真实edge"'
    )
    ax.text(12.75, 6.3, diff_text, fontproperties=FONT_CN_LIGHT,
            fontsize=10, color=GRAY, ha='center', va='top',
            linespacing=1.7, family='monospace')

    add_page_num(ax, 4)
    fig.savefig(OUT / 'slide_04_innovation.png', dpi=DPI, bbox_inches='tight',
                facecolor=BG, pad_inches=0.3)
    plt.close()


# ═══════════════════════════════════════════════════════════
# SLIDE 5: Agent Depth
# ═══════════════════════════════════════════════════════════
def slide_05_agent_depth():
    fig, ax = new_slide()
    add_top_bar(ax, 'BTC-Pulse', 'Agent使用深度 25%')

    ax.text(8, 8.0, '四层Agent协作体系',
            fontproperties=FONT_CN_BOLD, fontsize=28, color=WHITE, ha='center')

    layers_data = [
        ('Layer 1  开发智能体', 'Coder ↔ Reviewer 对抗迭代\n评分≥9/10才通过，共7轮', ACCENT3, 7.0),
        ('Layer 2  分析智能体', 'LLM情感 + XGBoost预测 + IF异常\n多信号加权融合', ACCENT2, 5.5),
        ('Layer 3  决策智能体', 'Kelly仓位 + ATR止损 + 回撤熔断\nOptuna贝叶斯优化', ACCENT, 4.0),
        ('Layer 4  验证智能体', 'Bootstrap CI + DSR校正 + MC基准\n三重统计把关', ACCENT4, 2.5),
    ]

    for label, desc, color, y in layers_data:
        draw_card(ax, 1.0, y - 0.5, 6.5, 1.2, BG_CARD)
        ax.text(1.3, y + 0.2, label, fontproperties=FONT_CN_BOLD,
                fontsize=12, color=color, va='center')
        ax.text(1.3, y - 0.25, desc, fontproperties=FONT_CN_LIGHT,
                fontsize=9, color=GRAY, va='top', linespacing=1.4)

    # Arrows between layers
    for y in [6.5, 5.0, 3.5]:
        ax.annotate('', xy=(4.25, y - 0.05), xytext=(4.25, y + 0.45),
                    arrowprops=dict(arrowstyle='->', color=DIM, lw=1))

    # Right side: key point
    draw_card(ax, 8.5, 1.5, 6.5, 6.0, BG_CARD)
    ax.text(11.75, 7.1, '深度而非浅层', fontproperties=FONT_CN_BOLD,
            fontsize=15, color=ACCENT, ha='center')

    key_points = [
        ('x  浅层使用', '"让AI写一段代码"', ACCENT3),
        ('', '', WHITE),
        ('v  深层使用', '', ACCENT),
        ('', '• Agent间对抗协作（Coder vs Reviewer）', WHITE),
        ('', '• 多模型融合决策（Rules + ML + LLM）', WHITE),
        ('', '• 统计验证闭环（不盲信任何单一输出）', WHITE),
        ('', '• 全流程Agent覆盖', WHITE),
        ('', '  （开发→分析→决策→验证）', WHITE),
    ]

    y_pos = 6.3
    for title, desc, color in key_points:
        if title:
            ax.text(9.0, y_pos, title, fontproperties=FONT_CN_BOLD,
                    fontsize=11, color=color, va='center')
            y_pos -= 0.15
        if desc:
            ax.text(9.0, y_pos, desc, fontproperties=FONT_CN_LIGHT,
                    fontsize=10, color=GRAY if color != ACCENT3 else DIM, va='top')
        y_pos -= 0.5

    add_page_num(ax, 5)
    fig.savefig(OUT / 'slide_05_agent_depth.png', dpi=DPI, bbox_inches='tight',
                facecolor=BG, pad_inches=0.3)
    plt.close()


# ═══════════════════════════════════════════════════════════
# SLIDE 6: Practical Value
# ═══════════════════════════════════════════════════════════
def slide_06_practical_value():
    fig, ax = new_slide()
    add_top_bar(ax, 'BTC-Pulse', '实用价值 25%')

    ax.text(8, 8.0, '解决真实问题，具备孵化潜力',
            fontproperties=FONT_CN_BOLD, fontsize=28, color=WHITE, ha='center')

    # Left: real problems solved
    draw_card(ax, 0.5, 1.0, 7.2, 6.2, BG_CARD)
    ax.text(4.1, 6.8, '解决的真实问题', fontproperties=FONT_CN_BOLD,
            fontsize=16, color=ACCENT, ha='center')

    solved = [
        ('◆', '零成本数据', '全部免费公开API，无需付费数据源'),
        ('◆', '一键部署', 'pip install + run_ingest + run_dashboard'),
        ('◆', '回测可信', '交易成本+防前瞻+统计检验=真实绩效'),
        ('◆', '智能融合', '规则可解释 + ML泛化 + LLM语义理解'),
        ('◆', '参数优化', 'Optuna自动搜索 + DSR防过拟合校正'),
    ]

    for i, (icon, title, desc) in enumerate(solved):
        y = 5.9 - i * 1.05
        ax.text(1.2, y, icon, fontsize=16, va='center')
        ax.text(2.0, y, title, fontproperties=FONT_CN_BOLD,
                fontsize=12, color=WHITE, va='center')
        ax.text(3.8, y, desc, fontproperties=FONT_CN_LIGHT,
                fontsize=10, color=GRAY, va='center')

    # Right: incubation potential
    draw_card(ax, 8.3, 1.0, 7.2, 6.2, BG_CARD)
    ax.text(11.9, 6.8, '孵化潜力', fontproperties=FONT_CN_BOLD,
            fontsize=16, color=ACCENT4, ha='center')

    potential = [
        ('SaaS化', '→  订阅制信号推送服务（Telegram/Discord）'),
        ('多币种', '→  扩展至ETH/SOL，跨资产相关性信号'),
        ('API化', '→  REST API对外输出信号，集成到交易平台'),
        ('教育', '→  量化投资教学平台（从入门到回测验证）'),
        ('社区', '→  策略市场：用户上传/共享/评分策略'),
    ]

    for i, (title, desc) in enumerate(potential):
        y = 5.9 - i * 1.05
        ax.text(8.8, y, title, fontproperties=FONT_CN_BOLD,
                fontsize=12, color=ACCENT4, va='center')
        ax.text(10.4, y, desc, fontproperties=FONT_CN_LIGHT,
                fontsize=10, color=GRAY, va='center')

    add_page_num(ax, 6)
    fig.savefig(OUT / 'slide_06_practical_value.png', dpi=DPI, bbox_inches='tight',
                facecolor=BG, pad_inches=0.3)
    plt.close()


# ═══════════════════════════════════════════════════════════
# SLIDE 7: Signal Pipeline
# ═══════════════════════════════════════════════════════════
def slide_07_signal_pipeline():
    fig, ax = new_slide()
    add_top_bar(ax, 'BTC-Pulse', '核心功能演示')

    ax.text(8, 8.0, '复合信号生成流水线',
            fontproperties=FONT_CN_BOLD, fontsize=28, color=WHITE, ha='center')

    # Data sources (left)
    sources = [
        ('价格/成交量', ACCENT2, 6.8),
        ('恐惧贪婪指数', ACCENT4, 5.8),
        ('资金费率/OI', ACCENT, 4.8),
        ('新闻标题', ACCENT5, 3.8),
    ]
    for name, color, y in sources:
        draw_card(ax, 0.5, y - 0.35, 2.5, 0.7, BG_CARD)
        ax.text(1.75, y, name, fontproperties=FONT_CN,
                fontsize=10, color=color, ha='center', va='center')
        ax.annotate('', xy=(3.3, y), xytext=(3.0, y),
                    arrowprops=dict(arrowstyle='->', color=DIM, lw=1))

    # Feature Engine
    draw_card(ax, 3.3, 3.3, 2.2, 4.2, BG_CARD2)
    ax.text(4.4, 7.0, '特征工程', fontproperties=FONT_CN_BOLD,
            fontsize=12, color=ACCENT, ha='center')
    ax.text(4.4, 6.2, 'RSI · BB · MACD\nADX · OBV\nVWAP偏离\n累计资金费率\n前向标签',
            fontproperties=FONT_CN_LIGHT, fontsize=8, color=GRAY,
            ha='center', va='top', linespacing=1.5)

    ax.annotate('', xy=(5.8, 5.4), xytext=(5.5, 5.4),
                arrowprops=dict(arrowstyle='->', color=DIM, lw=1))

    # Dual model
    draw_card(ax, 5.9, 5.5, 2.3, 1.6, BG_CARD)
    ax.text(7.05, 6.7, '规则引擎 60%', fontproperties=FONT_CN_BOLD,
            fontsize=10, color=ACCENT2, ha='center')
    ax.text(7.05, 6.1, '反向情绪·杠杆清洗\n机构基准', fontproperties=FONT_CN_LIGHT,
            fontsize=8, color=GRAY, ha='center', va='top', linespacing=1.3)

    draw_card(ax, 5.9, 3.5, 2.3, 1.6, BG_CARD)
    ax.text(7.05, 4.7, 'XGBoost 40%', fontproperties=FONT_CN_BOLD,
            fontsize=10, color=ACCENT5, ha='center')
    ax.text(7.05, 4.1, 'Walk-forward训练\n三分类概率', fontproperties=FONT_CN_LIGHT,
            fontsize=8, color=GRAY, ha='center', va='top', linespacing=1.3)

    # Merge arrow
    ax.annotate('', xy=(8.5, 5.3), xytext=(8.2, 6.0),
                arrowprops=dict(arrowstyle='->', color=DIM, lw=1))
    ax.annotate('', xy=(8.5, 5.3), xytext=(8.2, 4.6),
                arrowprops=dict(arrowstyle='->', color=DIM, lw=1))

    # Composite signal
    draw_card(ax, 8.5, 4.4, 2.5, 1.8, BG_CARD2)
    ax.text(9.75, 5.8, '复合信号', fontproperties=FONT_CN_BOLD,
            fontsize=13, color=ACCENT, ha='center')
    ax.text(9.75, 5.1, 'Strong Buy → Strong Sell\n五级评分输出',
            fontproperties=FONT_CN_LIGHT, fontsize=9, color=GRAY,
            ha='center', va='top', linespacing=1.4)

    ax.annotate('', xy=(11.3, 5.3), xytext=(11.0, 5.3),
                arrowprops=dict(arrowstyle='->', color=DIM, lw=1))

    # Validation
    draw_card(ax, 11.3, 3.3, 4.0, 4.2, BG_CARD)
    ax.text(13.3, 7.0, '统计验证层', fontproperties=FONT_CN_BOLD,
            fontsize=12, color=ACCENT4, ha='center')
    validation_items = [
        'Freqtrade回测引擎',
        '0.1%/边 手续费建模',
        '次根K线开盘入场',
        'Bootstrap 95% CI',
        '缩水Sharpe (DSR)',
        '随机入场MC p-value',
    ]
    for i, item in enumerate(validation_items):
        ax.text(11.7, 6.2 - i * 0.55, f'• {item}',
                fontproperties=FONT_CN_LIGHT, fontsize=9, color=GRAY)

    # Bottom annotation
    ax.text(8, 2.2, '每个信号经过  融合 → 回测 → 统计检验  三重验证，无单一Agent输出被直接信任',
            fontproperties=FONT_CN, fontsize=11, color=ACCENT, ha='center')

    add_page_num(ax, 7)
    fig.savefig(OUT / 'slide_07_signal_pipeline.png', dpi=DPI, bbox_inches='tight',
                facecolor=BG, pad_inches=0.3)
    plt.close()


# ═══════════════════════════════════════════════════════════
# SLIDE 8: Backtest Results Demo
# ═══════════════════════════════════════════════════════════
def slide_08_backtest_demo():
    fig, ax = new_slide()
    add_top_bar(ax, 'BTC-Pulse', 'Demo: 回测输出示例')

    ax.text(8, 8.0, '回测报告：不止收益率',
            fontproperties=FONT_CN_BOLD, fontsize=28, color=WHITE, ha='center')

    # Left: metrics card
    draw_card(ax, 0.8, 1.2, 6.5, 6.0, BG_CARD)
    ax.text(4.05, 6.8, '绩效指标', fontproperties=FONT_CN_BOLD,
            fontsize=14, color=ACCENT, ha='center')

    metrics = [
        ('总交易数', '47', WHITE),
        ('胜率', '63.8%', ACCENT),
        ('胜率 95% CI', '[51.2%, 75.4%]', GRAY),
        ('总收益', '+42.3%', ACCENT),
        ('Sharpe', '1.34', ACCENT2),
        ('Sharpe 95% CI', '[0.87, 1.81]', GRAY),
        ('最大回撤', '-12.4%', ACCENT3),
        ('Calmar', '2.15', ACCENT2),
        ('手续费总计', '$847.20', ACCENT4),
    ]

    for i, (label, value, color) in enumerate(metrics):
        y = 6.1 - i * 0.55
        ax.text(1.2, y, label, fontproperties=FONT_CN_LIGHT,
                fontsize=10, color=GRAY, va='center')
        ax.text(6.8, y, value, fontproperties=FONT_CN_BOLD,
                fontsize=10, color=color, va='center', ha='right')

    # Right: statistical validation card
    draw_card(ax, 8.0, 1.2, 7.2, 6.0, BG_CARD)
    ax.text(11.6, 6.8, '统计验证（本项目独有）',
            fontproperties=FONT_CN_BOLD, fontsize=14, color=ACCENT4, ha='center')

    stats = [
        ('缩水Sharpe (DSR)', '0.972', '> 0.95 = PASS', ACCENT),
        ('随机基准 p-value', '0.023', '< 0.05 = Significant', ACCENT),
        ('平均收益 CI', '[0.8%, 2.1%]', '不含零 = Significant', ACCENT),
    ]

    for i, (label, value, interpret, color) in enumerate(stats):
        y = 5.8 - i * 1.3
        ax.text(8.5, y + 0.2, label, fontproperties=FONT_CN_BOLD,
                fontsize=12, color=WHITE, va='center')
        ax.text(8.5, y - 0.25, value, fontproperties=FONT_CN_BOLD,
                fontsize=16, color=color, va='center')
        ax.text(12.0, y - 0.25, interpret, fontproperties=FONT_CN_LIGHT,
                fontsize=10, color=GRAY, va='center')

    # Bottom: vs others
    draw_card(ax, 8.3, 1.5, 6.6, 1.2, BG_CARD2)
    ax.text(11.6, 2.4, '对比：传统回测只报告收益率和Sharpe\n本系统额外提供3项统计显著性检验，防止自欺欺人',
            fontproperties=FONT_CN_LIGHT, fontsize=9, color=DIM,
            ha='center', va='top', linespacing=1.5)

    add_page_num(ax, 8)
    fig.savefig(OUT / 'slide_08_backtest_demo.png', dpi=DPI, bbox_inches='tight',
                facecolor=BG, pad_inches=0.3)
    plt.close()


# ═══════════════════════════════════════════════════════════
# SLIDE 9: Tech Stack
# ═══════════════════════════════════════════════════════════
def slide_09_tech_stack():
    fig, ax = new_slide()
    add_top_bar(ax, 'BTC-Pulse', '技术栈')

    ax.text(8, 8.0, '技术基础与工具链',
            fontproperties=FONT_CN_BOLD, fontsize=28, color=WHITE, ha='center')

    categories = [
        ('数据层', ['CCXT (交易所API)', 'Bitcoinity CSV', 'feedparser (RSS)', 'SQLite (WAL)'], ACCENT2),
        ('计算层', ['Pandas / NumPy', 'SciPy (统计)', 'XGBoost (ML)', 'scikit-learn (IF)'], ACCENT),
        ('优化层', ['Optuna (贝叶斯)', 'Bootstrap (10K)', 'DSR (Bailey 2014)', 'Monte Carlo'], ACCENT4),
        ('应用层', ['Streamlit (UI)', 'Loguru (日志)', 'pytest (23测试)', 'Git (版本控制)'], ACCENT5),
    ]

    for i, (cat, items, color) in enumerate(categories):
        x = 0.8 + i * 3.85
        draw_card(ax, x, 2.0, 3.5, 5.2, BG_CARD)
        ax.text(x + 1.75, 6.8, cat, fontproperties=FONT_CN_BOLD,
                fontsize=14, color=color, ha='center')
        for j, item in enumerate(items):
            y = 5.8 - j * 0.9
            ax.text(x + 0.4, y, f'> {item}', fontproperties=FONT_CN_LIGHT,
                    fontsize=10, color=GRAY)

    # Bottom
    ax.text(8, 1.2, '全栈 Python 实现  ·  100% 免费数据源  ·  MIT 开源协议',
            fontproperties=FONT_CN, fontsize=12, color=DIM, ha='center')

    add_page_num(ax, 9)
    fig.savefig(OUT / 'slide_09_tech_stack.png', dpi=DPI, bbox_inches='tight',
                facecolor=BG, pad_inches=0.3)
    plt.close()


# ═══════════════════════════════════════════════════════════
# SLIDE 10: Privacy & Compliance
# ═══════════════════════════════════════════════════════════
def slide_10_privacy():
    fig, ax = new_slide()
    add_top_bar(ax, 'BTC-Pulse', '隐私合规 10%')

    ax.text(8, 8.0, '隐私合规：零PII设计',
            fontproperties=FONT_CN_BOLD, fontsize=28, color=WHITE, ha='center')

    # Principles
    draw_card(ax, 0.5, 4.5, 7.0, 2.9, BG_CARD)
    ax.text(4.0, 7.0, '设计原则', fontproperties=FONT_CN_BOLD,
            fontsize=14, color=ACCENT, ha='center')

    principles = [
        ('数据最小化', '仅采集公开市场数据，零用户个人信息'),
        ('本地优先', '全部存储在本地SQLite，不上传云端'),
        ('密钥隔离', 'config.yaml在.gitignore，不进版本控制'),
        ('可审计', '全部数据来源、处理流程代码可追溯'),
    ]
    for i, (title, desc) in enumerate(principles):
        y = 6.4 - i * 0.55
        ax.text(1.0, y, f'• {title}', fontproperties=FONT_CN_BOLD,
                fontsize=10, color=WHITE, va='center')
        ax.text(3.5, y, desc, fontproperties=FONT_CN_LIGHT,
                fontsize=9, color=GRAY, va='center')

    # Risk handling
    draw_card(ax, 8.0, 4.5, 7.5, 2.9, BG_CARD)
    ax.text(11.75, 7.0, '风险识别与应对', fontproperties=FONT_CN_BOLD,
            fontsize=14, color=ACCENT4, ha='center')

    risks = [
        ('LLM API传输', '仅传公开新闻标题，支持本地Ollama', '[OK]'),
        ('API Key泄露', '.gitignore + example模板 + 环境变量', '[OK]'),
        ('数据库暴露', '本地文件 + 表名白名单防注入', '[OK]'),
        ('日志敏感信息', 'Loguru不记录密钥或个人信息', '[OK]'),
    ]
    for i, (risk, handling, status) in enumerate(risks):
        y = 6.4 - i * 0.55
        ax.text(8.4, y, risk, fontproperties=FONT_CN_BOLD,
                fontsize=9, color=ACCENT3, va='center')
        ax.text(10.5, y, handling, fontproperties=FONT_CN_LIGHT,
                fontsize=9, color=GRAY, va='center')
        ax.text(15.0, y, status, fontsize=10, va='center', ha='center')

    # Data classification table
    draw_card(ax, 0.5, 1.0, 15.0, 2.8, BG_CARD2)
    ax.text(8.0, 3.4, '数据分类表', fontproperties=FONT_CN_BOLD,
            fontsize=13, color=WHITE, ha='center')

    headers = ['数据类型', '来源', '涉及PII', '合规状态']
    for i, h in enumerate(headers):
        ax.text(1.5 + i * 3.7, 2.9, h, fontproperties=FONT_CN_BOLD,
                fontsize=9, color=ACCENT, va='center')

    rows = [
        ('BTC价格/成交量', 'Bitcoinity/CCXT', '[NO] 否', '[OK] 公开市场数据'),
        ('恐惧贪婪指数', 'Alternative.me', '[NO] 否', '[OK] 聚合统计指标'),
        ('衍生品数据', '交易所公开API', '[NO] 否', '[OK] 匿名聚合数据'),
        ('新闻标题', 'RSS公开源', '[NO] 否', '[OK] 已公开内容'),
    ]
    for j, row in enumerate(rows):
        y = 2.4 - j * 0.4
        for i, cell in enumerate(row):
            ax.text(1.5 + i * 3.7, y, cell, fontproperties=FONT_CN_LIGHT,
                    fontsize=9, color=GRAY, va='center')

    add_page_num(ax, 10)
    fig.savefig(OUT / 'slide_10_privacy.png', dpi=DPI, bbox_inches='tight',
                facecolor=BG, pad_inches=0.3)
    plt.close()


# ═══════════════════════════════════════════════════════════
# SLIDE 11: Roadmap
# ═══════════════════════════════════════════════════════════
def slide_11_roadmap():
    fig, ax = new_slide()
    add_top_bar(ax, 'BTC-Pulse', '发展路线')

    ax.text(8, 8.0, '演进路线图',
            fontproperties=FONT_CN_BOLD, fontsize=28, color=WHITE, ha='center')

    phases = [
        ('已完成', [
            '多源数据采集框架',
            '复合信号引擎 (Rules+ML)',
            'Freqtrade风格回测引擎',
            '统计验证层 (DSR/CI/MC)',
            'Streamlit仪表盘',
            'Coder+Reviewer双Agent开发',
        ], ACCENT, 0.5),
        ('短期目标', [
            'Telegram/Discord信号推送',
            'FreqAI自适应滚动重训练',
            'Docker一键部署',
            'Paper Trading前向测试',
        ], ACCENT2, 5.5),
        ('长期愿景', [
            '多币种(ETH/SOL)扩展',
            'REST API信号输出',
            '强化学习交易Agent',
            '社区策略市场',
        ], ACCENT4, 10.5),
    ]

    for title, items, color, x in phases:
        draw_card(ax, x, 1.5, 4.5, 5.8, BG_CARD)
        ax.text(x + 2.25, 6.9, title, fontproperties=FONT_CN_BOLD,
                fontsize=14, color=color, ha='center')
        for j, item in enumerate(items):
            y = 6.0 - j * 0.7
            ax.text(x + 0.5, y, f'> {item}', fontproperties=FONT_CN_LIGHT,
                    fontsize=10, color=GRAY)

    # Arrows
    for x in [5.0, 10.0]:
        ax.annotate('', xy=(x + 0.8, 4.4), xytext=(x, 4.4),
                    arrowprops=dict(arrowstyle='->', color=DIM, lw=1.5))

    add_page_num(ax, 11)
    fig.savefig(OUT / 'slide_11_roadmap.png', dpi=DPI, bbox_inches='tight',
                facecolor=BG, pad_inches=0.3)
    plt.close()


# ═══════════════════════════════════════════════════════════
# SLIDE 12: Summary / Thank You
# ═══════════════════════════════════════════════════════════
def slide_12_summary():
    fig, ax = new_slide()

    ax.text(8, 7.5, 'BTC-Pulse',
            fontproperties=FONT_CN_BOLD, fontsize=42, color=WHITE, ha='center')

    ax.plot([3, 13], [6.8, 6.8], color=ACCENT, linewidth=2, alpha=0.6)

    ax.text(8, 6.2, '从"看起来赚钱"到"统计上显著"',
            fontproperties=FONT_CN, fontsize=18, color=ACCENT, ha='center')

    # Five dimensions summary
    dims = [
        ('创新性', '多维融合 + 三重统计验证闭环', ACCENT2),
        ('Agent深度', '四层Agent协作（开发→分析→决策→验证）', ACCENT5),
        ('实用价值', '零成本部署 · 真实回测 · 可孵化SaaS', ACCENT),
        ('展示表达', '完整Demo · 架构图 · 回测报告', ACCENT4),
        ('隐私合规', '零PII设计 · 本地优先 · 密钥隔离', ACCENT3),
    ]

    for i, (dim, desc, color) in enumerate(dims):
        y = 5.0 - i * 0.8
        ax.text(4.0, y, dim, fontproperties=FONT_CN_BOLD,
                fontsize=13, color=color, ha='right', va='center')
        ax.text(4.4, y, desc, fontproperties=FONT_CN_LIGHT,
                fontsize=12, color=GRAY, ha='left', va='center')

    ax.text(8, 1.5, 'github.com/Ilovecodinghhh/BTC-Pulse',
            fontproperties=FONT_CN_LIGHT, fontsize=13, color=ACCENT2, ha='center')
    ax.text(8, 0.9, 'Thank You  ·  谢谢',
            fontproperties=FONT_CN, fontsize=16, color=WHITE, ha='center')

    add_page_num(ax, 12)
    fig.savefig(OUT / 'slide_12_summary.png', dpi=DPI, bbox_inches='tight',
                facecolor=BG, pad_inches=0.3)
    plt.close()


# ═══════════════════════════════════════════════════════════
# Generate all slides
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating slides...")
    slide_01_title()
    print("  [1/12] Title")
    slide_02_problem()
    print("  [2/12] Problem")
    slide_03_architecture()
    print("  [3/12] Architecture")
    slide_04_innovation()
    print("  [4/12] Innovation")
    slide_05_agent_depth()
    print("  [5/12] Agent Depth")
    slide_06_practical_value()
    print("  [6/12] Practical Value")
    slide_07_signal_pipeline()
    print("  [7/12] Signal Pipeline")
    slide_08_backtest_demo()
    print("  [8/12] Backtest Demo")
    slide_09_tech_stack()
    print("  [9/12] Tech Stack")
    slide_10_privacy()
    print("  [10/12] Privacy")
    slide_11_roadmap()
    print("  [11/12] Roadmap")
    slide_12_summary()
    print("  [12/12] Summary")
    print(f"\nAll 12 slides saved to: {OUT}/")
