# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ハンズオン1: サプライヤー選定問題
#
# **— JijModelingで最適化問題を"自分で実装"しよう —**
#
# ## 業務シナリオ
#
# あなたは製造業の調達部門の担当者です。
# 新製品を組み立てるために **30種類の部品** を調達する必要があります。
# 部品は4カテゴリ（電子部品・機構部品・素材/化学品・外装/梱包）に分かれ、
# **10社のサプライヤー候補** がそれぞれ異なる価格を提示しています。
#
# ただし、各サプライヤーには供給能力の上限があり、**1社あたり最大5部品まで** しか対応できません。
# さらに、同じサプライヤーから複数部品をまとめ買いすると **セット割引** が適用されます。
#
# **30部品 × 10社 = 300個のバイナリ変数** — 手計算では絶対に解けない、
# 本格的な組合せ最適化問題に挑みましょう。
#
# - **Step 1**: 供給能力制約のあるコスト最小化（一次式）
# - **Step 2**: セット割引（二次項）を導入して、DA/イジングマシンの真価を体感
# - **Step 3**: 富士通デジタルアニーラで解いてみよう
# - **Step 4**: JijZept Solverで解いてみよう

# %%
import jijmodeling as jm
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # .env からAPIトークンを読み込み
print(f"JijModeling version: {jm.__version__}")

# %% [markdown]
# ---
# ## Step 1: サプライヤー選定（供給能力制約つき）
#
# ### データの準備
#
# 30部品を4カテゴリに分類し、10社のサプライヤーにはそれぞれ得意カテゴリがあります。
# コストは「ベース価格 × 得意度 × ばらつき」で生成します（シード固定で再現可能）。

# %%
rng = np.random.default_rng(seed=42)

# --- 4カテゴリ × 部品 ---
categories = ["電子部品", "機構部品", "素材・化学品", "外装・梱包"]
parts_per_cat = {
    "電子部品":    ["モーター", "基板", "センサー", "ディスプレイ",
                    "コントローラ", "バッテリー", "コネクタ", "LED"],
    "機構部品":    ["外装", "フレーム", "冷却ファン", "ギア",
                    "ベアリング", "スプリング", "シャフト"],
    "素材・化学品": ["接着剤", "コーティング", "絶縁材", "ゴムシール",
                    "放熱グリス", "はんだ", "ケーブル", "樹脂材"],
    "外装・梱包":  ["カバー", "パネル", "ラベル", "緩衝材",
                    "段ボール", "取扱説明書", "ネジセット"],
}

# フラットリスト構築
part_names = []
part_cat_idx = []   # 各部品のカテゴリindex (0-3)
for ci, cat in enumerate(categories):
    for part in parts_per_cat[cat]:
        part_names.append(part)
        part_cat_idx.append(ci)

NUM_P = len(part_names)   # 30
assert NUM_P == 30

supplier_names = [f"サプライヤー{chr(65 + i)}" for i in range(10)]
NUM_S = 10
CAPACITY = 5  # 各サプライヤーの供給上限

# --- コストテーブル生成 ---
# カテゴリ別ベース価格（万円）
base_costs = np.array([150, 80, 40, 30])  # 電子 / 機構 / 素材 / 外装

# サプライヤー得意度 (値が小さいほど安い)
#   0.70 = 非常に得意, 0.85 = やや得意, 1.0 = 普通
spec = np.ones((NUM_S, 4))
spec[0, 0] = 0.75; spec[0, 1] = 0.90   # A: 電子に強い
spec[1, 0] = 0.80; spec[1, 2] = 0.85   # B: 電子/素材
spec[2, 1] = 0.70; spec[2, 0] = 0.90   # C: 機構に強い
spec[3, 1] = 0.75; spec[3, 3] = 0.85   # D: 機構/外装
spec[4, 2] = 0.70; spec[4, 1] = 0.90   # E: 素材に強い
spec[5, 2] = 0.75; spec[5, 3] = 0.85   # F: 素材/外装
spec[6, 3] = 0.70; spec[6, 2] = 0.90   # G: 外装に強い
spec[7, 3] = 0.75; spec[7, 0] = 0.85   # H: 外装/電子
spec[8, 0] = 0.72; spec[8, 2] = 0.88   # I: 電子に強い(攻め)
spec[9, 1] = 0.72; spec[9, 3] = 0.88   # J: 機構に強い(攻め)

cost_data = np.zeros((NUM_P, NUM_S), dtype=int)
for pi in range(NUM_P):
    ci = part_cat_idx[pi]
    base = base_costs[ci]
    for si in range(NUM_S):
        noise = rng.uniform(0.88, 1.12)
        cost_data[pi, si] = int(round(base * spec[si, ci] * noise))

print(f"【コストテーブル生成完了】")
print(f"  部品数:       {NUM_P}（{' / '.join(f'{c}:{len(parts_per_cat[c])}' for c in categories)}）")
print(f"  サプライヤー数: {NUM_S}")
print(f"  供給上限:      {CAPACITY}部品/社（10社×5 = 50枠 で30部品を割り当て）")
print(f"  コスト範囲:    {cost_data.min()}〜{cost_data.max()}万円")
print(f"  バイナリ変数:  {NUM_P} × {NUM_S} = {NUM_P * NUM_S}個")

# 各部品の最安サプライヤーを確認
cheapest_per_part = cost_data.argmin(axis=1)
parts_at_cheapest = np.bincount(cheapest_per_part, minlength=NUM_S)
over_cap = [(supplier_names[si], int(parts_at_cheapest[si]))
            for si in range(NUM_S) if parts_at_cheapest[si] > CAPACITY]
if over_cap:
    print(f"\n  ⚠ 制約なしで最安を選ぶと供給上限を超えるサプライヤー:")
    for name, cnt in over_cap:
        print(f"    {name}: {cnt}部品（上限{CAPACITY}）→ 単純な最安選択は不可能！")

# %%
import matplotlib
import matplotlib.pyplot as plt

# 可視化用の英語ラベル
part_labels = [
    "Motor", "PCB", "Sensor", "Display", "Controller", "Battery", "Connector", "LED",
    "Housing", "Frame", "Fan", "Gear", "Bearing", "Spring", "Shaft",
    "Adhesive", "Coating", "Insulation", "Seal", "ThermalPaste", "Solder", "Cable", "Resin",
    "Cover", "Panel", "Label", "Cushion", "Cardboard", "Manual", "Screws",
]
supplier_labels = [f"Sup.{chr(65 + i)}" for i in range(10)]
category_labels = ["Electronics", "Mechanical", "Materials", "Packaging"]
cat_colors = ["#e74c3c", "#3498db", "#2ecc71", "#e67e22"]

# カテゴリ区切り線の位置
cat_boundaries = []
pos = 0
for cat in categories:
    pos += len(parts_per_cat[cat])
    cat_boundaries.append(pos)

# コストテーブルヒートマップ
fig, ax = plt.subplots(figsize=(10, 12))
im = ax.imshow(cost_data, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(NUM_S))
ax.set_xticklabels(supplier_labels, fontsize=9)
ax.set_yticks(range(NUM_P))
ax.set_yticklabels(part_labels, fontsize=8)

# カテゴリ区切り線
for b in cat_boundaries[:-1]:
    ax.axhline(y=b - 0.5, color="black", linewidth=2)

# カテゴリラベル（右側）
start = 0
for ci, cat in enumerate(categories):
    n = len(parts_per_cat[cat])
    mid = start + n / 2 - 0.5
    ax.text(NUM_S + 0.3, mid, category_labels[ci], va="center", fontsize=9,
            fontweight="bold", color=cat_colors[ci])
    start += n

fig.colorbar(im, ax=ax, label="Cost (10K JPY)", shrink=0.6)
ax.set_title(f"Cost Table: {NUM_P} Parts x {NUM_S} Suppliers", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### JijModeling 2 でモデルを構築
#
# #### 定式化
#
# **決定変数**: $x_{p,s} \in \{0, 1\}$ — 部品 $p$ をサプライヤー $s$ から調達するか
#
# **目的関数（最小化）**:
# $$\min \sum_{p=0}^{29} \sum_{s=0}^{9} c_{p,s} \cdot x_{p,s}$$
#
# **制約条件1** — 各部品は必ず1社から調達:
# $$\sum_{s} x_{p,s} = 1 \quad \forall p \in \{0, \ldots, 29\}$$
#
# **制約条件2** — 各サプライヤーの供給上限:
# $$\sum_{p} x_{p,s} \leq K \quad \forall s \in \{0, \ldots, 9\}$$

""# %%
@jm.Problem.define("supplier_selection_step1", sense=jm.ProblemSense.MINIMIZE)
def problem_step1(problem: jm.DecoratedProblem):
    P = problem.Natural(description="部品数")
    S = problem.Natural(description="サプライヤー数")
    K = problem.Natural(description="供給上限")
    c = problem.Float(shape=(P, S), description="コストテーブル")

    x = problem.BinaryVar(shape=(P, S), description="調達選択")

    # 目的関数: 総コストを最小化
    problem += jm.sum(c[p, s] * x[p, s] for p in P for s in S)

    # 制約1: 各部品は1社から調達
    problem += problem.Constraint(
        "one_supplier_per_part",
        [jm.sum(x[p, s] for s in S) == 1 for p in P],
        description="各部品は1社から調達",
    )

    # 制約2: 各サプライヤーの供給上限
    problem += problem.Constraint(
        "supplier_capacity",
        [jm.sum(x[p, s] for p in P) <= K for s in S],
        description="各サプライヤーの供給上限",
    )

print("【完成したモデル】")
print(problem_step1)

# %% [markdown]
# ### JijModelingの基本要素
#
# | 要素 | JM2 コード | 役割 |
# |------|-----------|------|
# | パラメータ（スカラー） | `problem.Natural()` | 部品数(30), サプライヤー数(10), 供給上限(5) |
# | パラメータ（行列） | `problem.Float(shape=(P, S))` | コストテーブル(30×10) |
# | 決定変数 | `problem.BinaryVar(shape=(P, S))` | 300個の0/1変数 |
# | 目的関数 | `problem += jm.sum(...)` | 総コスト最小化 |
# | 等式制約 | `Constraint([... == 1 ...])` | 各部品は1社から |
# | 不等式制約 | `Constraint([... <= K ...])` | 供給上限 |
#
# **ポイント**: 300変数の問題でも、コード量は10行程度。
# **数式をそのまま書く** のがJijModelingの強みです。

# %% [markdown]
# ### 求解: OpenJij（シミュレーテッドアニーリング）

# %%
from ommx_openjij_adapter import OMMXOpenJijSAAdapter

instance_data_step1 = {
    "P": NUM_P, "S": NUM_S, "K": CAPACITY, "c": cost_data,
}
instance_step1 = problem_step1.eval(instance_data_step1)

sample_set_step1 = OMMXOpenJijSAAdapter.sample(
    instance_step1,
    num_reads=500,
    num_sweeps=3000,
    uniform_penalty_weight=1000.0,
)
best_step1 = sample_set_step1.best_feasible_unrelaxed

# %% [markdown]
# ### 結果の確認

# %%
def show_result_summary(best, cost_matrix, label=""):
    """結果をサプライヤー別にまとめて表示"""
    df = best.decision_variables_df
    sel = df[(df["name"] == "x") & (df["value"] == 1.0)]

    assignments = {}
    for _, row in sel.iterrows():
        pi, si = row["subscripts"]
        assignments[pi] = si

    # サプライヤー別集計
    supplier_parts = [[] for _ in range(NUM_S)]
    for pi, si in sorted(assignments.items()):
        supplier_parts[si].append(pi)

    total_cost = 0
    for si in range(NUM_S):
        if not supplier_parts[si]:
            continue
        parts = supplier_parts[si]
        cost_sum = sum(cost_matrix[pi][si] for pi in parts)
        total_cost += cost_sum
        part_str = ", ".join(part_names[pi] for pi in parts)
        bar = "█" * len(parts) + "░" * (CAPACITY - len(parts))
        print(f"  {supplier_names[si]:12s} [{bar}] {len(parts)}部品 {cost_sum:>5}万円  {part_str}")

    print(f"\n  → 総コスト: {total_cost}万円")
    return total_cost, assignments


print(f"【最適解 Step1: 供給能力制約つき最安調達（OpenJij SA）】\n")
total_cost_step1, assignments_step1 = show_result_summary(best_step1, cost_data)

# 参考値
ref_total = int(cost_data.min(axis=1).sum())
print(f"  （参考: 制約なし最安合計 = {ref_total}万円 ← 供給上限を超えるため不可）")

# %%
# 結果可視化: サプライヤー別の部品数（カテゴリ色分け）
fig, ax = plt.subplots(figsize=(10, 5))

bottom = np.zeros(NUM_S)
for ci, cat in enumerate(categories):
    counts = np.zeros(NUM_S)
    for pi, si in assignments_step1.items():
        if part_cat_idx[pi] == ci:
            counts[si] += 1
    ax.bar(supplier_labels, counts, bottom=bottom, label=category_labels[ci],
           color=cat_colors[ci], edgecolor="white", width=0.7)
    bottom += counts

ax.axhline(y=CAPACITY, color="red", linestyle="--", linewidth=1.5, label=f"Capacity ({CAPACITY})")
ax.set_ylabel("Number of Parts", fontsize=12)
ax.set_title(f"Step 1: Supplier Allocation (Total: {total_cost_step1})", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, CAPACITY + 2)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Step 2: セット割引（二次項）を導入！
#
# ### 追加条件
#
# 同じサプライヤーから複数の部品をまとめて買うと「セット割引」が適用されます。
# 各サプライヤーが得意カテゴリ内で3ペアずつ、**合計30通りの割引** を提示。
#
# この「2つの部品の組合せ」で割引が決まるので、目的関数に **二次項** が登場。
# 供給能力制約 × 30通りの割引 → 手計算は完全に不可能です。

# %%
discount_data = np.zeros((NUM_P, NUM_P, NUM_S))

# サプライヤーA (電子部品に強い)
discount_data[0, 1, 0] = 30   # モーター + 基板
discount_data[2, 5, 0] = 40   # センサー + バッテリー
discount_data[3, 7, 0] = 25   # ディスプレイ + LED

# サプライヤーB (電子/素材)
discount_data[1, 4, 1] = 35   # 基板 + コントローラ
discount_data[0, 6, 1] = 28   # モーター + コネクタ
discount_data[5, 7, 1] = 20   # バッテリー + LED

# サプライヤーC (機構部品に強い)
discount_data[8, 9, 2] = 30   # 外装 + フレーム
discount_data[10, 12, 2] = 25  # 冷却ファン + ベアリング
discount_data[11, 14, 2] = 20  # ギア + シャフト

# サプライヤーD (機構/外装)
discount_data[8, 10, 3] = 35  # 外装 + 冷却ファン
discount_data[9, 13, 3] = 25  # フレーム + スプリング
discount_data[12, 14, 3] = 30  # ベアリング + シャフト

# サプライヤーE (素材に強い)
discount_data[15, 17, 4] = 25  # 接着剤 + 絶縁材
discount_data[16, 20, 4] = 35  # コーティング + はんだ
discount_data[18, 22, 4] = 20  # ゴムシール + 樹脂材

# サプライヤーF (素材/外装)
discount_data[15, 16, 5] = 30  # 接着剤 + コーティング
discount_data[19, 21, 5] = 25  # 放熱グリス + ケーブル
discount_data[20, 22, 5] = 35  # はんだ + 樹脂材

# サプライヤーG (外装に強い)
discount_data[23, 24, 6] = 40  # カバー + パネル
discount_data[25, 27, 6] = 25  # ラベル + 段ボール
discount_data[26, 29, 6] = 20  # 緩衝材 + ネジセット

# サプライヤーH (外装/電子)
discount_data[23, 25, 7] = 35  # カバー + ラベル
discount_data[24, 28, 7] = 22  # パネル + 取扱説明書
discount_data[27, 29, 7] = 28  # 段ボール + ネジセット

# サプライヤーI (電子部品・攻めの価格)
discount_data[0, 5, 8] = 50   # モーター + バッテリー（最大割引！）
discount_data[2, 4, 8] = 35   # センサー + コントローラ
discount_data[1, 7, 8] = 25   # 基板 + LED

# サプライヤーJ (機構部品・攻めの価格)
discount_data[8, 11, 9] = 35  # 外装 + ギア
discount_data[9, 14, 9] = 28  # フレーム + シャフト
discount_data[10, 13, 9] = 32  # 冷却ファン + スプリング

print("【セット割引テーブル — 30ペア（各社3ペア）】\n")
total_available = 0
for si in range(NUM_S):
    pairs = []
    for p1 in range(NUM_P):
        for p2 in range(p1 + 1, NUM_P):
            d = discount_data[p1, p2, si]
            if d > 0:
                pairs.append((p1, p2, d))
                total_available += d
    if pairs:
        print(f"  {supplier_names[si]}:")
        for p1, p2, d in pairs:
            print(f"    {part_names[p1]} + {part_names[p2]}: -{d:.0f}万円")

print(f"\n  合計{len([1 for p1 in range(NUM_P) for p2 in range(p1+1,NUM_P) for s in range(NUM_S) if discount_data[p1,p2,s]>0])}ペア, "
      f"割引総額（全適用時）: {total_available:.0f}万円")

# %%
# 割引可視化: サプライヤー別の利用可能割引額
fig, ax = plt.subplots(figsize=(10, 4))
disc_per_supplier = []
for si in range(NUM_S):
    total = sum(discount_data[p1, p2, si]
                for p1 in range(NUM_P) for p2 in range(p1 + 1, NUM_P))
    disc_per_supplier.append(total)

sup_colors = ["#e74c3c", "#e74c3c", "#3498db", "#3498db", "#2ecc71",
              "#2ecc71", "#e67e22", "#e67e22", "#e74c3c", "#3498db"]
bars = ax.bar(supplier_labels, disc_per_supplier, color=sup_colors,
              edgecolor="white", width=0.7, alpha=0.8)
for bar, val in zip(bars, disc_per_supplier):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.0f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Available Discount (10K JPY)", fontsize=11)
ax.set_title("Discount Budget by Supplier (3 pairs each)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### JijModelingでモデルを記述（二次項つき）
#
# **目的関数**: 総コスト − セット割引 を最小化
#
# $$\min \sum_{p,s} c_{p,s} \cdot x_{p,s} - \sum_{p_1 < p_2} \sum_{s} d_{p_1, p_2, s} \cdot x_{p_1, s} \cdot x_{p_2, s}$$
#
# $x_{p_1,s} \cdot x_{p_2,s}$ が **二次項**。同じサプライヤー $s$ を選んだ時にのみ割引が発生。
#
# **制約**: Step1と同じ（各部品1社 + 供給上限）

# %%
@jm.Problem.define("supplier_selection_with_discount", sense=jm.ProblemSense.MINIMIZE)
def problem_step2(problem: jm.DecoratedProblem):
    P = problem.Natural(description="部品数")
    S = problem.Natural(description="サプライヤー数")
    K = problem.Natural(description="供給上限")
    c = problem.Float(shape=(P, S), description="コストテーブル")
    d = problem.Float(shape=(P, P, S), description="セット割引テーブル")
    x = problem.BinaryVar(shape=(P, S), description="調達選択")

    cost = jm.sum(c[p, s] * x[p, s] for p in P for s in S)
    discount = jm.sum(
        d[p1, p2, s] * x[p1, s] * x[p2, s]
        for p1 in P for p2 in P for s in S
        if p1 < p2
    )
    problem += cost - discount

    problem += problem.Constraint(
        "one_supplier_per_part",
        [jm.sum(x[p, s] for s in S) == 1 for p in P],
        description="各部品は1社から調達",
    )
    problem += problem.Constraint(
        "supplier_capacity",
        [jm.sum(x[p, s] for p in P) <= K for s in S],
        description="各サプライヤーの供給上限",
    )

print("【定式化結果（二次項つき）】")
print(problem_step2)

# %% [markdown]
# ### 求解（OpenJij SA）

# %%
instance_data_step2 = {
    "P": NUM_P, "S": NUM_S, "K": CAPACITY,
    "c": cost_data, "d": discount_data,
}
instance_step2 = problem_step2.eval(instance_data_step2)

sample_set_step2 = OMMXOpenJijSAAdapter.sample(
    instance_step2,
    num_reads=500,
    num_sweeps=3000,
    uniform_penalty_weight=1000.0,
)
best_step2 = sample_set_step2.best_feasible_unrelaxed

# %% [markdown]
# ### 結果の確認 + Step1 との比較

# %%
def show_step2_result(best, label=""):
    """Step2の結果表示（割引含む）"""
    df = best.decision_variables_df
    sel = df[(df["name"] == "x") & (df["value"] == 1.0)]

    assignments = {}
    for _, row in sel.iterrows():
        pi, si = row["subscripts"]
        assignments[pi] = si

    # サプライヤー別集計
    supplier_parts = [[] for _ in range(NUM_S)]
    for pi, si in sorted(assignments.items()):
        supplier_parts[si].append(pi)

    base_cost = 0
    for si in range(NUM_S):
        if not supplier_parts[si]:
            continue
        parts = supplier_parts[si]
        cost_sum = sum(cost_data[pi][si] for pi in parts)
        base_cost += cost_sum
        bar = "█" * len(parts) + "░" * (CAPACITY - len(parts))
        print(f"  {supplier_names[si]:12s} [{bar}] {len(parts)}部品 {cost_sum:>5}万円")

    # 割引計算
    total_discount = 0
    applied_discounts = []
    for p1 in range(NUM_P):
        for p2 in range(p1 + 1, NUM_P):
            if assignments.get(p1) == assignments.get(p2):
                si = assignments[p1]
                disc = discount_data[p1, p2, si]
                if disc > 0:
                    total_discount += disc
                    applied_discounts.append((p1, p2, si, disc))

    if applied_discounts:
        print(f"\n  ★ 適用された割引:")
        for p1, p2, si, disc in applied_discounts:
            print(f"    {part_names[p1]} + {part_names[p2]} @{supplier_names[si]}: -{disc:.0f}万円")

    net_cost = base_cost - total_discount
    print(f"\n  基本コスト: {base_cost}万円")
    print(f"  セット割引: -{total_discount:.0f}万円")
    print(f"  → 実質コスト: {net_cost:.0f}万円")
    return net_cost, assignments


print("【最適解 Step2: セット割引考慮（OpenJij SA）】\n")
net_cost_oj, assignments_step2 = show_step2_result(best_step2)

print(f"\n【比較】")
print(f"  Step1（単純最安）: {total_cost_step1:.0f}万円")
print(f"  Step2（割引考慮）: {net_cost_oj:.0f}万円")
if net_cost_oj < total_cost_step1:
    print(f"  → 割引による追加削減: {total_cost_step1 - net_cost_oj:.0f}万円 お得！")

# %%
# Step1 vs Step2: サプライヤー割当の比較
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, assigns, title in [
    (axes[0], assignments_step1, f"Step 1: Min Cost ({total_cost_step1})"),
    (axes[1], assignments_step2, f"Step 2: With Discount ({net_cost_oj:.0f})"),
]:
    bottom = np.zeros(NUM_S)
    for ci, cat in enumerate(categories):
        counts = np.zeros(NUM_S)
        for pi, si in assigns.items():
            if part_cat_idx[pi] == ci:
                counts[si] += 1
        ax.bar(supplier_labels, counts, bottom=bottom, label=category_labels[ci],
               color=cat_colors[ci], edgecolor="white", width=0.7)
        bottom += counts

    ax.axhline(y=CAPACITY, color="red", linestyle="--", linewidth=1.5)
    ax.set_ylabel("Parts", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, CAPACITY + 2)
    ax.legend(fontsize=8, loc="upper right")

fig.suptitle("Step 1 vs Step 2: Supplier Allocation Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Step 3: 富士通デジタルアニーラ（DA）で解いてみよう
#
# ここまでは手元のPC上のソルバー（OpenJij SA）で解きました。
# 次には、**クラウド上の高性能ソルバー** として富士通デジタルアニーラ（DA）を使ってみましょう。
#
# **富士通デジタルアニーラ（DA）** — 専用ハードウェアによる高速・高精度求解を体験します。
#
# ### JijModelingの強み: ソルバー差し替えが簡単
#
# モデル（Problem）は一切変更せず、**求解部分だけ差し替える** だけです。

# %%
import os
from ommx_da4_adapter import OMMXDA4Adapter

DA_TOKEN = os.environ.get("DA_API_TOKEN", "")
if not DA_TOKEN:
    print("⚠ 環境変数 DA_API_TOKEN が設定されていません。")
    print("  export DA_API_TOKEN='your-token-here'")

# %% [markdown]
# ### Step1をDAで解く

# %%
if DA_TOKEN:
    print("【Step1: 供給能力制約つき最安調達 — Digital Annealer v3c】\n")

    sample_set_da1 = OMMXDA4Adapter.sample(
        instance_step1, token=DA_TOKEN, version="v3c"
    )
    best_da1 = sample_set_da1.best_feasible_unrelaxed
    total_da1, _ = show_result_summary(best_da1, cost_data, "DA v3c")

    print(f"\n  参考: OpenJij SA = {total_cost_step1:.0f}万円")

# %% [markdown]
# ### Step2（セット割引つき）をDAで解く

# %%
if DA_TOKEN:
    print("【Step2: セット割引考慮 — Digital Annealer v3c】\n")

    sample_set_da2 = OMMXDA4Adapter.sample(
        instance_step2, token=DA_TOKEN, version="v3c"
    )
    best_da2 = sample_set_da2.best_feasible_unrelaxed
    net_cost_da, _ = show_step2_result(best_da2)

    print(f"\n  参考: OpenJij SA = {net_cost_oj:.0f}万円")

# %% [markdown]
# ### （参考）DA v4 でも解いてみる

# %%
if DA_TOKEN:
    print("【Step2: セット割引考慮 — Digital Annealer v4】\n")

    sample_set_da2_v4 = OMMXDA4Adapter.sample(
        instance_step2, token=DA_TOKEN, version="v4"
    )
    best_da2_v4 = sample_set_da2_v4.best_feasible_unrelaxed
    net_cost_da_v4, _ = show_step2_result(best_da2_v4)

# %% [markdown]
# ---
# ## Step 4: JijZept Solver で解いてみよう
#
# **JijZept Solver** は Jij が提供する汎用最適化ソルバーです。
# QUBO/イジング形式だけでなく、**連続変数や不等式制約を含む一般的な数理最適化問題**にも対応します。

# %%
import jijzept_solver

JIJZEPT_TOKEN = os.environ.get("JIJZEPT_SOLVER_ACCESS_TOKEN", "")
if not JIJZEPT_TOKEN:
    print("⚠ 環境変数 JIJZEPT_SOLVER_ACCESS_TOKEN が設定されていません。")
    print("  export JIJZEPT_SOLVER_ACCESS_TOKEN='your-token-here'")

# %% [markdown]
# ### Step2 を JijZept Solver で解く

# %%
if JIJZEPT_TOKEN:
    print("【Step2: セット割引考慮 — JijZept Solver】\n")

    solution_jz = jijzept_solver.solve(instance_step2, solve_limit_sec=10.0)
    net_cost_jz, _ = show_step2_result(solution_jz)
else:
    net_cost_jz = None

# %% [markdown]
# ---
# ## 全ソルバー比較

# %%
print("【Step2 全ソルバー比較】\n")
print(f"  {'ソルバー':<24s}  {'実質コスト':>10s}")
print(f"  {'-'*38}")
print(f"  {'OpenJij SA':<24s}  {net_cost_oj:>10.0f}万円")
if DA_TOKEN:
    print(f"  {'Digital Annealer v3c':<24s}  {net_cost_da:>10.0f}万円")
    print(f"  {'Digital Annealer v4':<24s}  {net_cost_da_v4:>10.0f}万円")
if net_cost_jz is not None:
    print(f"  {'JijZept Solver':<24s}  {net_cost_jz:>10.0f}万円")

# %%
solver_results = [("OpenJij SA", net_cost_oj)]
if DA_TOKEN:
    solver_results.append(("DA v3c", net_cost_da))
    solver_results.append(("DA v4", net_cost_da_v4))
if net_cost_jz is not None:
    solver_results.append(("JijZept Solver", net_cost_jz))

solver_names_list = [r[0] for r in solver_results]
solver_costs = [r[1] for r in solver_results]
bar_colors_solver = ["#3498db", "#e74c3c", "#e67e22", "#2ecc71"][:len(solver_results)]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(solver_names_list, solver_costs, color=bar_colors_solver,
               edgecolor="white", height=0.5)
for bar, cost in zip(bars, solver_costs):
    ax.text(bar.get_width() + max(solver_costs) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{cost:.0f}", va="center", fontsize=11, fontweight="bold")
ax.set_xlabel("Net Cost (10K JPY)", fontsize=12)
ax.set_title("Step 2: Solver Comparison — Net Cost (300 variables)", fontsize=14, fontweight="bold")
ax.set_xlim(0, max(solver_costs) * 1.15)
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## まとめ
#
# ### 今回の問題スケール
#
# | 項目 | 値 |
# |------|------|
# | 部品数 | 30（4カテゴリ） |
# | サプライヤー数 | 10 |
# | バイナリ変数 | **300個** |
# | 制約: 各部品1社 | 30本（等式） |
# | 制約: 供給上限 | 10本（不等式） |
# | セット割引ペア | 30通り |
# | 全探索の場合 | $10^{30}$ 通り → 現実的に不可能 |
#
# ### Step1 vs Step2 の違い
#
# | | Step1（一次式 + 容量制約） | Step2（二次項 + 容量制約） |
# |---|---|---|
# | 目的関数 | $\sum c \cdot x$（線形） | $\sum c \cdot x - \sum d \cdot x \cdot x$（二次） |
# | 手計算 | 容量制約で組合せ的に困難 | 割引の相互作用でさらに困難 |
# | DA/イジングマシン | 制約つき最適化を高速求解 | **真価を発揮！** |
#
# ### ソルバーの使い分け
#
# | ソルバー | 特徴 | 得意な問題 |
# |---------|------|-----------|
# | **OpenJij** | 手元PCで動く、OSS | 小規模問題の開発・テスト |
# | **富士通デジタルアニーラ** | 専用HW、100Kビット対応 | **大規模QUBO問題**（離散変数・制約が少ない） |
# | **JijZept Solver** | 汎用数理最適化ソルバー | **連続変数・制約が多い問題**（混合整数計画等） |
#
# #### どう選ぶ？
#
# ```
# 問題の性質を確認
#   ├── 変数が全てバイナリ（0/1）で、制約が等式のみ → DA が得意！
#   │     └── 大規模（変数1,000個〜）ならDAの専用HWが圧倒的に速い
#   ├── 連続変数が混ざる or 不等式制約が多い → JijZept Solver が得意！
#   │     └── 一般的な数理最適化問題（線形計画、混合整数計画など）
#   └── 小規模テスト・プロトタイピング → OpenJij（ローカルで即実行）
# ```
#
# ### 実務スケール
#
# - 今回: 30部品 × 10社 = 300変数（ハンズオンサイズ）
# - 実務: 部品500個 × サプライヤー50社 = 25,000変数
# - 全探索は $50^{500}$ 通り → 宇宙の年齢でも終わらない
# - → **デジタルアニーラなら数秒〜数分で良い解を発見！**
#
# ### モデルとデータの分離（OMMX）
#
# 数式（Problem）とデータ（instance_data）が分かれている
# → **データを差し替えるだけで別の問題が解ける**
# → **ソルバーも差し替え可能**（OpenJij → DA → JijZept Solver → 量子アニーラ）
#
# すべてのソルバーが **OMMX Instance** という共通フォーマットで接続されているため、
# モデルを書き直す必要が一切ありません。
#
# ### 次のステップ: ハンズオン2へ
#
# もっと複雑な問題（倉庫レイアウト最適化）を
# **JijZept AIに自然言語で伝えて** モデルを自動生成してみましょう！
