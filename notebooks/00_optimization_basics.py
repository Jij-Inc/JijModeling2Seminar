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
# # 数理最適化の基礎 — JijModelingで学ぶ最適化入門
#
# **— 講義パート（20分）：最適化の基本概念からJijModelingまで —**
#
# ## このノートブックで学ぶこと
#
# 1. **数理最適化の基礎用語** — 決定変数・目的関数・制約条件
# 2. **QUBOとは** — デジタルアニーラが解く形式
# 3. **ペナルティ法** — 制約をQUBOに組み込む方法
# 4. **JijModeling 2 と OMMX** — 数式をそのままコードにする

# %% [markdown]
# ---
# ## Section 1: 数理最適化の基礎用語
#
# ### 数理最適化とは？
#
# 「与えられた条件の下で、最も良い答えを見つける」ための数学的な手法です。
#
# $$\min_x f(x) \quad \text{s.t.} \quad g_j(x) \leq C_j$$
#
# | 用語 | 意味 | 例 |
# |------|------|-----|
# | **決定変数** $x$ | 何を決めるか（選択・割当） | 「このアイテムを買うか？」(0 or 1) |
# | **目的関数** $f(x)$ | 何を最適化するか | 「合計価値を最大化」 |
# | **制約条件** $g(x) \leq C$ | 守るべきルール | 「予算500万円以内」 |

# %% [markdown]
# ### 具体例：アイテム選択問題
#
# **シナリオ**: IT機器の調達担当として、予算500万円で4つのアイテムから価値が最大になる組合せを選びたい。
#
# | アイテム | 価値 | コスト |
# |----------|------|--------|
# | A: 高性能サーバー | 10 | 300万円 |
# | B: ネットワーク機器 | 7 | 200万円 |
# | C: セキュリティソフト | 5 | 100万円 |
# | D: モニター | 3 | 150万円 |
#
# **定式化**:
#
# - **決定変数**: $x_i \in \{0, 1\}$ — アイテム $i$ を選ぶ(1) / 選ばない(0)
# - **目的関数**: $\max\ 10x_A + 7x_B + 5x_C + 3x_D$
# - **制約条件**: $300x_A + 200x_B + 100x_C + 150x_D \leq 500$
#
# 4変数 → $2^4 = 16$ 通り。全列挙できるサイズですが、
# 変数が増えると **組合せ爆発** が起きます（30変数で約10億通り、300変数で $2^{300}$ 通り）。

# %%
import itertools
import numpy as np

# アイテムデータ
items = ["A: 高性能サーバー", "B: ネットワーク機器", "C: セキュリティソフト", "D: モニター"]
values = np.array([10, 7, 5, 3])
costs = np.array([300, 200, 100, 150])  # 万円
budget = 500  # 万円

# 全16通りをブルートフォースで列挙
print("【全16通りの列挙】\n")
print(f"  {'選択':>12s}  {'価値':>4s}  {'コスト':>6s}  {'判定'}")
print(f"  {'-'*40}")

best_value = -1
best_combo = None

for bits in itertools.product([0, 1], repeat=4):
    x = np.array(bits)
    total_value = int(values @ x)
    total_cost = int(costs @ x)
    feasible = total_cost <= budget

    # 選択されたアイテムの表示
    selected = [chr(65 + i) for i in range(4) if x[i] == 1]
    label = ",".join(selected) if selected else "(なし)"
    mark = ""
    if feasible and total_value > best_value:
        best_value = total_value
        best_combo = x.copy()
    if feasible:
        mark = " ★" if total_value == best_value and np.array_equal(x, best_combo) else " ○"
    else:
        mark = " ✗ 予算超過"

    print(f"  {label:>12s}  {total_value:>4d}  {total_cost:>5d}万円 {mark}")

print(f"\n  → 最適解: {','.join(chr(65+i) for i in range(4) if best_combo[i]==1)}")
print(f"    価値={best_value}, コスト={int(costs @ best_combo)}万円")

# %% [markdown]
# ### Problem → Model → Instance
#
# 最適化では、以下の3つのレベルを区別します。
#
# | レベル | 意味 | 例 |
# |--------|------|-----|
# | **Problem（問題）** | 抽象的な問題の種類 | 「予算制約つきアイテム選択」 |
# | **Model（モデル）** | 数式で書いた定式化 | $\max \sum v_i x_i,\ \text{s.t.}\ \sum c_i x_i \leq B$ |
# | **Instance（インスタンス）** | 具体的な数値を入れたもの | 価値=[10,7,5,3], コスト=[300,200,100,150], B=500 |
#
# JijModelingでは **Model（数式）** と **Instance（データ）** を分離して扱います。
# → データを差し替えるだけで別の問題が解ける！

# %% [markdown]
# ---
# ## Section 2: QUBO とは
#
# ### QUBO の定義
#
# **QUBO** = **Q**uadratic **U**nconstrained **B**inary **O**ptimization
#
# $$\min_x \sum_{i,j} Q_{ij} x_i x_j \quad (x_i \in \{0, 1\})$$
#
# | 単語 | 意味 |
# |------|------|
# | **Quadratic** | 二次式（$x_i \cdot x_j$ の積が登場） |
# | **Unconstrained** | 制約なし（制約はペナルティとして目的関数に組み込む） |
# | **Binary** | 変数は 0 か 1 |
# | **Optimization** | 最小化問題 |
#
# デジタルアニーラ（DA）やイジングマシンは、この **QUBO形式** を直接解くことができます。

# %% [markdown]
# ### Q行列の具体例
#
# アイテム選択の **目的関数部分** をQ行列で表現してみましょう（制約はSection 3で扱います）。
#
# 目的関数: $\max\ 10x_A + 7x_B + 5x_C + 3x_D$
#
# QUBOは最小化なので符号を反転: $\min\ -10x_A - 7x_B - 5x_C - 3x_D$
#
# バイナリ変数の性質: $x_i^2 = x_i$（0²=0, 1²=1）なので、一次項を対角成分に置けます:
#
# $$Q = \begin{pmatrix} -10 & 0 & 0 & 0 \\ 0 & -7 & 0 & 0 \\ 0 & 0 & -5 & 0 \\ 0 & 0 & 0 & -3 \end{pmatrix}$$
#
# $x^T Q x = -10x_A^2 - 7x_B^2 - 5x_C^2 - 3x_D^2 = -10x_A - 7x_B - 5x_C - 3x_D$

# %%
# Q行列で目的関数を表現（最小化のため符号反転）
Q_obj = np.diag([-10, -7, -5, -3]).astype(float)
print("【Q行列（目的関数部分）】")
print(Q_obj)

# 検算: いくつかの解で x @ Q @ x を計算
print("\n【検算: x^T Q x の値】")
test_cases = [
    ([1, 0, 1, 0], "A,C"),
    ([1, 1, 0, 0], "A,B"),
    ([0, 1, 1, 1], "B,C,D"),
    ([1, 1, 1, 1], "全選択"),
]
for bits, label in test_cases:
    x = np.array(bits, dtype=float)
    obj = x @ Q_obj @ x
    print(f"  {label:>6s}: x^T Q x = {obj:>5.0f}  （価値 = {-obj:.0f}）")

# %% [markdown]
# ---
# ## Section 3: QUBOへの変換 — ペナルティ法
#
# ### ペナルティ法の考え方
#
# QUBOは「制約なし」（Unconstrained）の形式ですが、実際の問題には制約があります。
# **ペナルティ法** で制約を目的関数に組み込みます:
#
# $$\min\ f(x) + \lambda \cdot (\text{制約違反の度合い})^2$$
#
# - $\lambda$（ペナルティ係数）: 十分大きな正の値
# - 制約を満たす解 → ペナルティ = 0（影響なし）
# - 制約を破る解 → ペナルティが大きくなり、その解は選ばれなくなる
#
# | 制約の種類 | ペナルティの形 |
# |-----------|--------------|
# | 等式制約: $h(x) = 0$ | $\lambda \cdot h(x)^2$ |
# | 不等式制約: $g(x) \leq C$ | スラック変数を導入して等式に変換 |

# %% [markdown]
# ### 予算制約のペナルティ化
#
# 予算制約: $300x_A + 200x_B + 100x_C + 150x_D \leq 500$
#
# 簡単のため、不等式制約をペナルティ化する代わりに
# 「予算を超えた分の二乗」でペナルティを与えます:
#
# $$P(x) = \lambda \cdot \max(0,\ 300x_A + 200x_B + 100x_C + 150x_D - 500)^2$$
#
# 完全なQUBO目的関数:
#
# $$\min\ \underbrace{-10x_A - 7x_B - 5x_C - 3x_D}_{\text{価値（符号反転）}} + \underbrace{\lambda \cdot \max(0,\ \text{コスト} - 500)^2}_{\text{予算超過ペナルティ}}$$

# %%
# ペナルティ法の効果を数値で確認
lam = 0.01  # ペナルティ係数（コストが万円単位なので小さめに設定）

print("【ペナルティ法の効果】\n")
print(f"  ペナルティ係数 λ = {lam}")
print(f"  {'選択':>10s}  {'価値':>4s}  {'コスト':>6s}  {'超過':>6s}  {'ペナルティ':>10s}  {'QUBO値':>10s}")
print(f"  {'-'*62}")

examples = [
    ([1, 0, 1, 0], "A,C"),    # 最適解: コスト400 ≤ 500
    ([1, 1, 0, 0], "A,B"),    # 実行可能: コスト500 ≤ 500
    ([0, 1, 1, 1], "B,C,D"),  # 実行可能: コスト450 ≤ 500
    ([1, 1, 1, 0], "A,B,C"),  # 制約違反: コスト600 > 500
    ([1, 1, 0, 1], "A,B,D"),  # 制約違反: コスト650 > 500
    ([1, 1, 1, 1], "全選択"),  # 制約違反: コスト750 > 500
]

for bits, label in examples:
    x = np.array(bits)
    value = int(values @ x)
    cost = int(costs @ x)
    excess = max(0, cost - budget)
    penalty = lam * excess ** 2
    qubo_val = -value + penalty
    feasible = "○" if excess == 0 else "✗"
    print(f"  {label:>10s}  {value:>4d}  {cost:>5d}万円  {excess:>5d}万円  {penalty:>10.1f}  {qubo_val:>10.1f}  {feasible}")

print(f"\n  → 制約違反解はペナルティにより QUBO値が大きくなり、最小化で自然に排除される")
print(f"  → 実行可能解の中で A,B（QUBO値 = -17.0）が最小 = 最適解！")

# %% [markdown]
# ---
# ## Section 4: JijModeling 2 と OMMX
#
# ### ワークフロー
#
# 手動でQ行列やペナルティを計算するのは大変です。
# **JijModeling** を使えば、数式をそのままPythonで書くだけ！
#
# ```
# ① Problem定義（数式を書く）
#     ↓
# ② Instance生成（具体的な数値を入れる）
#     ↓
# ③ Solve（ソルバーで解く）
#     ↓
# ④ 結果確認（最適解を取り出す）
# ```
#
# QUBO変換やペナルティ係数の設定は **ソルバーアダプタが自動で処理** してくれます。

# %%
import jijmodeling as jm

print(f"JijModeling version: {jm.__version__}")

# %%
# ① Problem定義: アイテム選択問題をJijModelingで記述
@jm.Problem.define("item_selection", sense=jm.ProblemSense.MAXIMIZE)
def item_problem(problem: jm.DecoratedProblem):
    N = problem.Natural(description="アイテム数")
    v = problem.Float(shape=(N,), description="各アイテムの価値")
    c = problem.Float(shape=(N,), description="各アイテムのコスト")
    B = problem.Float(description="予算上限")

    x = problem.BinaryVar(shape=(N,), description="選択変数")

    # 目的関数: 合計価値を最大化
    problem += jm.sum(v[i] * x[i] for i in N)

    # 制約条件: 予算以内
    problem += problem.Constraint(
        "budget",
        jm.sum(c[i] * x[i] for i in N) <= B,
        description="予算制約",
    )

print("【JijModelingで定義した数理モデル】")
print(item_problem)

# %%
# ② Instance生成: 具体的なデータを入れる
instance_data = {
    "N": 4,
    "v": [10, 7, 5, 3],
    "c": [300, 200, 100, 150],
    "B": 500,
}
instance = item_problem.eval(instance_data)
print("【インスタンス生成完了】")
print(f"  変数数: {len(instance.decision_variables)}")

# %%
# ③ Solve: OpenJijで求解
from ommx_openjij_adapter import OMMXOpenJijSAAdapter

sample_set = OMMXOpenJijSAAdapter.sample(
    instance,
    num_reads=100,
    num_sweeps=1000,
    uniform_penalty_weight=50.0,
)
best = sample_set.best_feasible_unrelaxed

# %%
# ④ 結果確認: 最適解を取り出す
df = best.decision_variables_df
selected = df[(df["name"] == "x") & (df["value"] == 1.0)]

print("【ソルバーの結果】\n")
total_value = 0
total_cost = 0
for _, row in selected.iterrows():
    idx = row["subscripts"][0]
    total_value += int(values[idx])
    total_cost += int(costs[idx])
    print(f"  {items[idx]}: 価値={values[idx]}, コスト={costs[idx]}万円")

print(f"\n  合計価値: {total_value}")
print(f"  合計コスト: {total_cost}万円")

# ブルートフォースとの照合
print(f"\n【ブルートフォース結果との照合】")
print(f"  ブルートフォース最適解: 価値={best_value}, コスト={int(costs @ best_combo)}万円")
print(f"  ソルバー最適解:         価値={total_value}, コスト={total_cost}万円")
if total_value == best_value:
    print(f"  → 一致！ソルバーが正しく最適解を見つけました。")
else:
    print(f"  → 不一致。ペナルティ係数やnum_readsの調整が必要かもしれません。")

# %% [markdown]
# ---
# ## まとめ
#
# ### 用語のおさらい
#
# | 用語 | 意味 |
# |------|------|
# | 決定変数 | 何を決めるか（0/1の選択） |
# | 目的関数 | 最大化 or 最小化したい値 |
# | 制約条件 | 守るべきルール |
# | QUBO | DA/イジングマシンが解く二次形式 |
# | ペナルティ法 | 制約を目的関数に組み込む手法 |
#
# ### ワークフロー
#
# | ステップ | JijModeling のコード |
# |---------|---------------------|
# | ① Problem定義 | `@jm.Problem.define(...)` + 数式を記述 |
# | ② Instance生成 | `problem.eval(data_dict)` |
# | ③ 求解 | `OMMXOpenJijSAAdapter.sample(instance, ...)` |
# | ④ 結果確認 | `best.decision_variables_df` |
#
# ### JijModeling の基本要素
#
# | 要素 | コード | 今回の使い方 |
# |------|--------|-------------|
# | スカラーパラメータ | `problem.Natural()` | アイテム数(4) |
# | 配列パラメータ | `problem.Float(shape=(N,))` | 価値・コスト |
# | バイナリ変数 | `problem.BinaryVar(shape=(N,))` | 選択変数(4個) |
# | 目的関数 | `problem += jm.sum(...)` | 価値の最大化 |
# | 制約条件 | `problem.Constraint(...)` | 予算制約 |

# %% [markdown]
# ### 次のステップ → `01_supplier_selection.py`
#
# 次のハンズオンでは、この基礎を使って **本格的な組合せ最適化問題** に挑みます:
#
# - 4変数 → **300変数**（30部品 × 10サプライヤー）
# - 1つの制約 → **40本の制約**（等式30本 + 不等式10本）
# - さらに **セット割引（二次項）** も導入
# - OpenJij → **デジタルアニーラ** → **JijZept Solver** でソルバー比較
#
# 手計算では絶対に解けない問題を、JijModelingで実装してみましょう！
