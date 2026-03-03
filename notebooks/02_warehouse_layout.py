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
# # ハンズオン2: JijZept AIで倉庫レイアウト最適化
#
# **— AIと共創しながら最適化問題を解こう —**
#
# ## 概要
#
# ハンズオン1では自分でJijModelingコードを書きました。
# ハンズオン2では、**JijZept AIに自然言語で問題を伝えて、AIにモデルを作ってもらいます**。
#
# まずはこのノートブックで「正解」となるモデルと結果を確認し、
# その後JijZept AIで同じ問題に取り組みます。
#
# ## 本ノートブックの構成
#
# 1. **データ準備** — 同時注文頻度・棚間距離を定義
# 2. **QAPモデル構築** — JijModelingで定式化
# 3. **OpenJij で求解** — ローカルSAで動作確認
# 4. **富士通デジタルアニーラで求解** — 専用HWでの高速求解
# 5. **JijZept Solver で求解** — 汎用ソルバーとの比較
# 6. **JijZept AI セクション** — 自然言語でモデルを構築する体験
#
# ## 業務シナリオ
#
# > あなたは物流センターのレイアウト担当者です。
# > 倉庫内に10個の棚があり、10種類の商品を配置したいと考えています。
# > 「一緒に注文されやすい商品」は「物理的に近い棚」に配置することで、
# > ピッキング作業員の移動距離を減らしたいのです。
#
# これは **二次割当問題（QAP: Quadratic Assignment Problem）** です。

# %%
import os
import jijmodeling as jm
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # .env からAPIトークンを読み込み
print(f"JijModeling version: {jm.__version__}")

# %% [markdown]
# ---
# ## データの準備

# %%
# 商品リスト
product_names = [
    "ビール", "おつまみ", "お米", "調味料", "冷凍食品",
    "洗剤", "シャンプー", "トイレットペーパー", "お菓子", "飲料水",
]
N = len(product_names)

# 同時注文頻度（回/月）: freq[p1][p2] = 商品p1とp2が一緒に注文される回数
freq_data = np.full((N, N), 5.0)  # デフォルト: 5回/月
np.fill_diagonal(freq_data, 0)     # 自分自身は0

# 高頻度ペアを設定（対称行列）
high_freq_pairs = [
    (0, 1, 85),   # ビール - おつまみ
    (0, 9, 40),   # ビール - 飲料水
    (1, 8, 55),   # おつまみ - お菓子
    (2, 3, 70),   # お米 - 調味料
    (2, 4, 30),   # お米 - 冷凍食品
    (3, 4, 45),   # 調味料 - 冷凍食品
    (5, 6, 90),   # 洗剤 - シャンプー
    (5, 7, 60),   # 洗剤 - トイレットペーパー
    (6, 7, 50),   # シャンプー - トイレットペーパー
    (8, 9, 65),   # お菓子 - 飲料水
]
for p1, p2, f in high_freq_pairs:
    freq_data[p1, p2] = f
    freq_data[p2, p1] = f

print("【同時注文頻度（高頻度ペアのみ表示）】")
for p1, p2, f in high_freq_pairs:
    print(f"  {product_names[p1]} - {product_names[p2]}: {f}回/月")

# %%
import matplotlib
import matplotlib.pyplot as plt

# 可視化用の英語ラベル
product_labels = [
    "Beer", "Snacks", "Rice", "Seasoning", "Frozen",
    "Detergent", "Shampoo", "Toilet Paper", "Sweets", "Water",
]

# 同時注文頻度ヒートマップ
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(freq_data, cmap="YlOrRd", aspect="equal")
ax.set_xticks(range(N))
ax.set_xticklabels(product_labels, fontsize=9, rotation=45, ha="right")
ax.set_yticks(range(N))
ax.set_yticklabels(product_labels, fontsize=9)
for i in range(N):
    for j in range(N):
        val = freq_data[i, j]
        color = "white" if val > 50 else "black"
        ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8, color=color)
fig.colorbar(im, ax=ax, label="Co-order Frequency (times/month)", shrink=0.8)
ax.set_title("Co-order Frequency Heatmap", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# 棚の配置（2×5の格子状）
#   [0] [1] [2] [3] [4]
#   [5] [6] [7] [8] [9]

shelf_positions = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
]

# 棚間距離（マンハッタン距離 × 3m）
UNIT = 3.0  # 隣接棚間の距離
dist_data = np.zeros((N, N))
for l1 in range(N):
    for l2 in range(N):
        r1, c1 = shelf_positions[l1]
        r2, c2 = shelf_positions[l2]
        dist_data[l1, l2] = (abs(r1 - r2) + abs(c1 - c2)) * UNIT

print("\n【棚の配置図】")
print("  [0] [1] [2] [3] [4]")
print("  [5] [6] [7] [8] [9]")
print(f"\n  隣接棚間の距離: {UNIT}m")

# %%
# 棚間距離ヒートマップ
fig, ax = plt.subplots(figsize=(7, 6))
shelf_labels = [f"Shelf {i}" for i in range(N)]
im = ax.imshow(dist_data, cmap="Blues", aspect="equal")
ax.set_xticks(range(N))
ax.set_xticklabels(shelf_labels, fontsize=9, rotation=45, ha="right")
ax.set_yticks(range(N))
ax.set_yticklabels(shelf_labels, fontsize=9)
for i in range(N):
    for j in range(N):
        ax.text(j, i, f"{dist_data[i, j]:.0f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, label="Distance (m)", shrink=0.8)
ax.set_title("Shelf Distance Heatmap", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## QAP の定式化（JijModeling 2）
#
# ### 数学的定式化
#
# **決定変数**: $x_{p, L} \in \{0, 1\}$ — 商品 $p$ を棚 $L$ に配置するか
#
# **目的関数（最小化）**:
# $$\min \sum_{p_1} \sum_{p_2} \sum_{L_1} \sum_{L_2} f_{p_1, p_2} \cdot d_{L_1, L_2} \cdot x_{p_1, L_1} \cdot x_{p_2, L_2}$$
#
# **制約条件**:
# - 各商品は1棚に配置: $\sum_{L} x_{p, L} = 1 \quad \forall p$
# - 各棚に1商品を配置: $\sum_{p} x_{p, L} = 1 \quad \forall L$

# %%
@jm.Problem.define("warehouse_layout_QAP", sense=jm.ProblemSense.MINIMIZE)
def qap_problem(problem: jm.DecoratedProblem):
    M = problem.Natural(description="商品数=棚数")
    f = problem.Float(shape=(M, M), description="同時注文頻度")
    d = problem.Float(shape=(M, M), description="棚間距離")
    x = problem.BinaryVar(shape=(M, M), description="配置 x[商品][棚]")

    # 目的関数: 頻度 × 距離 × 配置 の総和を最小化
    problem += jm.sum(
        f[p1, p2] * d[l1, l2] * x[p1, l1] * x[p2, l2]
        for p1 in M for p2 in M for l1 in M for l2 in M
    )

    # 制約1: 各商品は必ず1つの棚に配置
    problem += problem.Constraint(
        "one_product_per_shelf",
        [jm.sum(x[p, l] for l in M) == 1 for p in M],
        description="各商品は1棚に配置",
    )

    # 制約2: 各棚には必ず1つの商品を配置
    problem += problem.Constraint(
        "one_shelf_per_product",
        [jm.sum(x[p, l] for p in M) == 1 for l in M],
        description="各棚に1商品を配置",
    )

print("【QAPモデル】")
print(qap_problem)

# %%
# 結果表示ヘルパー関数
def show_layout(best, label=""):
    """QAPの結果を棚配置図として表示"""
    df = best.decision_variables_df
    selected = df[(df["name"] == "x") & (df["value"] == 1.0)]

    layout = {}
    for _, row in selected.iterrows():
        pi, li = row["subscripts"]
        layout[li] = pi

    print(f"  棚配置図{' (' + label + ')' if label else ''}:")
    for row_idx in range(2):
        line = "  "
        for col_idx in range(5):
            shelf_id = row_idx * 5 + col_idx
            if shelf_id in layout:
                prod_id = layout[shelf_id]
                line += f"[{product_names[prod_id]:^8s}]"
            else:
                line += f"[{'???':^8s}]"
        print(line)

    product_to_shelf = {pi: li for li, pi in layout.items()}
    print(f"\n  高頻度ペアの配置チェック:")
    for p1, p2, freq in sorted(high_freq_pairs, key=lambda x: -x[2]):
        l1 = product_to_shelf.get(p1, -1)
        l2 = product_to_shelf.get(p2, -1)
        d = dist_data[l1, l2] if l1 >= 0 and l2 >= 0 else float("inf")
        mark = "○" if d <= UNIT else "△" if d <= UNIT * 2 else "×"
        print(f"    {mark} {product_names[p1]:6s} - {product_names[p2]:10s}"
              f"  頻度:{freq:3.0f}回  距離:{d:.1f}m")

    print(f"\n  目的関数値: {best.objective:.1f}")
    return best.objective


def show_layout_fig(best, label=""):
    """QAPの結果を棚配置図としてmatplotlibで可視化"""
    df = best.decision_variables_df
    sel = df[(df["name"] == "x") & (df["value"] == 1.0)]

    layout = {}  # shelf_id -> product_id
    product_to_shelf = {}
    for _, row in sel.iterrows():
        pi, li = row["subscripts"]
        layout[li] = pi
        product_to_shelf[pi] = li

    fig, ax = plt.subplots(figsize=(12, 5))

    # 棚グリッド描画（2×5）
    cell_w, cell_h = 2.0, 1.5
    for shelf_id in range(N):
        row_idx = shelf_id // 5
        col_idx = shelf_id % 5
        x = col_idx * cell_w
        y = (1 - row_idx) * cell_h  # 上段が上

        prod_id = layout.get(shelf_id)
        color = "#dfe6e9" if prod_id is None else "#74b9ff"
        rect = plt.Rectangle((x, y), cell_w * 0.9, cell_h * 0.8,
                              facecolor=color, edgecolor="#2d3436", linewidth=2, zorder=2)
        ax.add_patch(rect)
        cx, cy = x + cell_w * 0.45, y + cell_h * 0.4
        ax.text(cx, cy + 0.15, product_labels[prod_id] if prod_id is not None else "?",
                ha="center", va="center", fontsize=10, fontweight="bold", zorder=3)
        ax.text(cx, cy - 0.25, f"Shelf {shelf_id}", ha="center", va="center",
                fontsize=8, color="#636e72", zorder=3)

    # 高頻度ペア間を線で結ぶ
    for p1, p2, freq in sorted(high_freq_pairs, key=lambda x: -x[2]):
        l1 = product_to_shelf.get(p1)
        l2 = product_to_shelf.get(p2)
        if l1 is None or l2 is None:
            continue
        r1, c1 = l1 // 5, l1 % 5
        r2, c2 = l2 // 5, l2 % 5
        x1 = c1 * cell_w + cell_w * 0.45
        y1 = (1 - r1) * cell_h + cell_h * 0.4
        x2 = c2 * cell_w + cell_w * 0.45
        y2 = (1 - r2) * cell_h + cell_h * 0.4
        d = dist_data[l1, l2]
        lw = max(1, freq / 20)
        alpha = max(0.3, min(1.0, 1.0 - d / 24))
        color = "#e74c3c" if d <= UNIT else "#e67e22" if d <= UNIT * 2 else "#95a5a6"
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=alpha, zorder=1)

    ax.set_xlim(-0.3, 5 * cell_w)
    ax.set_ylim(-0.3, 2 * cell_h + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    title = f"Layout Result{' (' + label + ')' if label else ''} — Objective: {best.objective:.1f}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### 求解: OpenJij（シミュレーテッドアニーリング）

# %%
from ommx_openjij_adapter import OMMXOpenJijSAAdapter

instance_data_qap = {"M": N, "f": freq_data, "d": dist_data}
instance_qap = qap_problem.eval(instance_data_qap)

sample_set_qap = OMMXOpenJijSAAdapter.sample(
    instance_qap,
    num_reads=100,
    num_sweeps=2000,
    uniform_penalty_weight=5000.0,
)

best_qap_oj = sample_set_qap.best_feasible_unrelaxed

# %%
print("【最適配置結果 — OpenJij SA】\n")
obj_oj = show_layout(best_qap_oj, "OpenJij SA")

# %%
show_layout_fig(best_qap_oj, "OpenJij SA")

# %% [markdown]
# ---
# ## 富士通デジタルアニーラ（DA）で解く
#
# 同じモデル・同じデータで、**ソルバーだけDAに差し替えて**求解します。
# 変数100個（10×10）程度の問題はDAの得意領域です。

# %%
from ommx_da4_adapter import OMMXDA4Adapter

DA_TOKEN = os.environ.get("DA_API_TOKEN", "")
if not DA_TOKEN:
    print("⚠ 環境変数 DA_API_TOKEN が設定されていません。")
    print("  export DA_API_TOKEN='your-token-here'")

# %% [markdown]
# ### DA v3c で求解

# %%
if DA_TOKEN:
    print("【最適配置結果 — Digital Annealer v3c】\n")

    sample_set_da_v3c = OMMXDA4Adapter.sample(
        instance_qap, token=DA_TOKEN, version="v3c"
    )
    best_qap_da_v3c = sample_set_da_v3c.best_feasible_unrelaxed
    obj_da_v3c = show_layout(best_qap_da_v3c, "DA v3c")
    show_layout_fig(best_qap_da_v3c, "DA v3c")

# %% [markdown]
# ### DA v4 で求解

# %%
if DA_TOKEN:
    print("【最適配置結果 — Digital Annealer v4】\n")

    sample_set_da_v4 = OMMXDA4Adapter.sample(
        instance_qap, token=DA_TOKEN, version="v4"
    )
    best_qap_da_v4 = sample_set_da_v4.best_feasible_unrelaxed
    obj_da_v4 = show_layout(best_qap_da_v4, "DA v4")
    show_layout_fig(best_qap_da_v4, "DA v4")

# %% [markdown]
# ---
# ## JijZept Solver で解く
#
# **JijZept Solver** は連続変数や複雑な制約を含む一般的な数理最適化問題にも対応する汎用ソルバーです。

# %%
import jijzept_solver

JIJZEPT_TOKEN = os.environ.get("JIJZEPT_SOLVER_ACCESS_TOKEN", "")
if not JIJZEPT_TOKEN:
    print("⚠ 環境変数 JIJZEPT_SOLVER_ACCESS_TOKEN が設定されていません。")

# %%
if JIJZEPT_TOKEN:
    print("【最適配置結果 — JijZept Solver】\n")

    solution_jz = jijzept_solver.solve(instance_qap, solve_limit_sec=5.0)
    obj_jz = show_layout(solution_jz, "JijZept Solver")
    show_layout_fig(solution_jz, "JijZept Solver")
else:
    obj_jz = None

# %% [markdown]
# ---
# ### 全ソルバー比較

# %%
print("【ソルバー比較】\n")
print(f"  {'ソルバー':<24s}  {'目的関数値':>10s}")
print(f"  {'-'*38}")
print(f"  {'OpenJij SA':<24s}  {obj_oj:>10.1f}")
if DA_TOKEN:
    print(f"  {'Digital Annealer v3c':<24s}  {obj_da_v3c:>10.1f}")
    print(f"  {'Digital Annealer v4':<24s}  {obj_da_v4:>10.1f}")
if obj_jz is not None:
    print(f"  {'JijZept Solver':<24s}  {obj_jz:>10.1f}")

results = [("OpenJij SA", obj_oj)]
if DA_TOKEN:
    results += [("DA v3c", obj_da_v3c), ("DA v4", obj_da_v4)]
if obj_jz is not None:
    results.append(("JijZept Solver", obj_jz))
best_solver = min(results, key=lambda x: x[1])
print(f"\n  → 最良: {best_solver[0]}（目的関数値 {best_solver[1]:.1f}）")

# %%
# ソルバー比較棒グラフ
solver_names_list = [r[0] for r in results]
solver_objs = [r[1] for r in results]
bar_colors = ["#3498db", "#e74c3c", "#e67e22", "#2ecc71"][:len(results)]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(solver_names_list, solver_objs, color=bar_colors, edgecolor="white", height=0.5)
for bar, obj in zip(bars, solver_objs):
    ax.text(bar.get_width() + max(solver_objs) * 0.02, bar.get_y() + bar.get_height() / 2,
            f"{obj:.1f}", va="center", fontsize=11, fontweight="bold")
ax.set_xlabel("Objective Value (lower is better)", fontsize=12)
ax.set_title("Solver Comparison — Objective Value", fontsize=14, fontweight="bold")
ax.set_xlim(0, max(solver_objs) * 1.15)
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## JijZept AI セクション
#
# ここからは **JijZept AI** を使って、同じ問題に自然言語で取り組みます。
#
# ### Step 1: JijZept AIに問題を入力する（10分）
#
# 以下のプロンプトをJijZept AIに入力してください：
#
# ```
# 倉庫内の商品配置を最適化する問題を解きたいです。
#
# 【条件】
# - 10個の商品と10個の棚があります
# - 各商品は必ず1つの棚に配置します
# - 各棚には必ず1つの商品が配置されます
# - 商品間の「同時注文頻度」のデータがあります
# - 棚間の「物理的距離」のデータがあります
#
# 【目的】
# 同時注文頻度が高い商品ペアができるだけ近い棚に配置されるように、
# 「同時注文頻度 × 棚間距離」の合計を最小化してください。
#
# 【数学的補足】
# - これは二次割当問題（QAP）です
# - 決定変数は x[p][L] ∈ {0,1}（商品pを棚Lに配置するか）
# - 制約は等式制約のみ（one-hot制約）
# ```
#
# #### 確認ポイント
# - JijZept AIが生成したJijModelingコードを確認
# - 変数定義、制約、目的関数がハンズオン1で学んだ要素と一致しているか
# - `BinaryVar`, `Float`, `Constraint`, `sum` の使い方を確認

# %% [markdown]
# ### Step 2: サンプルデータを入力する（5分）
#
# ```
# 以下のサンプルデータでモデルを実行してください。
#
# 【商品リスト】
# 0: ビール, 1: おつまみ, 2: お米, 3: 調味料, 4: 冷凍食品,
# 5: 洗剤, 6: シャンプー, 7: トイレットペーパー, 8: お菓子, 9: 飲料水
#
# 【同時注文頻度（回/月）】※対称行列の上三角のみ記載
# ビール-おつまみ: 85
# ビール-飲料水: 40
# おつまみ-お菓子: 55
# お米-調味料: 70
# お米-冷凍食品: 30
# 調味料-冷凍食品: 45
# 洗剤-シャンプー: 90
# 洗剤-トイレットペーパー: 60
# シャンプー-トイレットペーパー: 50
# お菓子-飲料水: 65
# その他のペア: 5
#
# 【棚間距離（メートル）】※棚は2×5の格子状に配置
# 棚の配置図:
#   [0] [1] [2] [3] [4]
#   [5] [6] [7] [8] [9]
#
# 隣接棚間の距離: 3m
# それ以外: マンハッタン距離 × 3m
# ```

# %% [markdown]
# ### Step 3: 条件を追加してみよう（20分）
#
# #### 追加条件A: 重量制約
#
# ```
# 追加条件があります。
# お米(2)と飲料水(9)は重いので、入口に近い棚（棚番号0と5）に限定してください。
# つまり x[2][0] + x[2][5] = 1 かつ x[9][0] + x[9][5] = 1 としてください。
# ```
#
# #### 追加条件B: 特定ペアの隣接
#
# ```
# さらに追加条件です。
# ビール(0)とおつまみ(1)は販促上の理由で「必ず隣接する棚」に配置してください。
# 隣接する棚のペアは: (0,1), (1,2), (2,3), (3,4), (5,6), (6,7), (7,8), (8,9), (0,5), (1,6), (2,7), (3,8), (4,9) です。
# ```
#
# #### 自由課題: 自分のアイデアで条件を追加
#
# 例:
# - 「冷凍食品は冷蔵庫に近い棚4と棚9に限定したい」
# - 「洗剤とお菓子は匂い移りを防ぐため離れた棚にしたい」
# - 「新商品のプロモーション用に棚0は空けておきたい」

# %% [markdown]
# ---
# ## まとめ: ハンズオン1 vs ハンズオン2
#
# | 項目 | ハンズオン1（手書き） | ハンズオン2（AI共創） |
# |------|----------------------|----------------------|
# | モデル構築 | Pythonコードを1行ずつ | 自然言語で記述 |
# | 変更・追加 | コードを修正 | 会話で追加指示 |
# | 学習コスト | JijModeling文法の理解が必要 | 業務知識があればOK |
# | カスタマイズ性 | 完全にコントロール可能 | AIの提案を元に調整 |
# | 向いているシーン | 定型問題の本番実装 | 新規問題の探索・プロトタイピング |
#
# ### 実務適用のコツ
#
# 1. **まずJijZept AIでプロトタイプ** → 自然言語で問題を伝えて「解けそうか」を素早く検証
# 2. **モデルの妥当性確認** → ハンズオン1で学んだ知識で、AIが生成したモデルの正しさを判断
# 3. **本番実装はJijModeling直書き** → 検証済みモデルをベースにコードを洗練
# 4. **ソルバー選択** → 問題の性質に応じて最適なソルバーを選択
#
# ### ソルバーの使い分け
#
# | ソルバー | 得意な問題 | 選ぶ場面 |
# |---------|-----------|---------|
# | **富士通デジタルアニーラ** | 大規模QUBO | 離散変数のみ、制約が少ない、変数が多い |
# | **JijZept Solver** | 汎用数理最適化 | 連続変数が混ざる、不等式制約が多い |
# | **OpenJij** | 小規模テスト | 開発中のプロトタイピング |
#
# ### DA × JijZept の価値
#
# ```
# 【従来】
# 業務課題 → (数週間) → 数学者がモデル化 → (数日) → 実装 → テスト → 本番
#
# 【JijZept AI + DA】
# 業務課題 → (数分) → AIがモデル生成 → (即時) → DA実行 → 結果確認
#          ↑ 対話で修正 ↑
#          業務担当者自身が主体的にモデルを改善できる
# ```
