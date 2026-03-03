# JijModeling 2 ハンズオン


## セットアップ

### 前提条件

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (推奨) または pip

### インストール

```bash
git clone <this-repo>
cd jm2-handson 

# uvを使う場合（推奨）
uv sync

# pipを使う場合
pip install -e .
```

### APIトークンの設定

ノートブック内で富士通デジタルアニーラやJijZept Solverを使うには、環境変数を設定してください。

```bash
# 富士通デジタルアニーラ
export DA_API_TOKEN='your-da-token-here'

# JijZept Solver
export JIJZEPT_SOLVER_ACCESS_TOKEN='your-jijzept-token-here'
```

> **注**: トークンがなくても OpenJij（ローカルSA）での求解は動作します。

### Jupyter Notebook として開く

```bash
# .py → .ipynb に変換
uv run jupytext --to notebook notebooks/01_supplier_selection.py
uv run jupytext --to notebook notebooks/02_warehouse_layout.py

# Jupyterを起動
uv run jupyter notebook notebooks/
```

### Python スクリプトとして直接実行

```bash
uv run python notebooks/01_supplier_selection.py
uv run python notebooks/02_warehouse_layout.py
```

## セミナー構成

| パート | 内容 | 時間 |
|--------|------|------|
| 講義 | 数理最適化の基本 + JijModelingとは | 20分 |
| **ハンズオン1** | サプライヤー選定問題（自分で実装） | 60分 |
| **ハンズオン2** | 倉庫レイアウト最適化（JijZept AI共創） | 60分 |

## ファイル構成

```
notebooks/
  01_supplier_selection.py   # ハンズオン1（jupytext %形式）
  02_warehouse_layout.py     # ハンズオン2（jupytext %形式）
docs/
  jm2_migration_notes.md     # JM1→JM2 移行メモ
  jijzept_ai_guide.md        # JijZept AI プロンプト集
  seminar_timetable.md       # 進行台本
```

## 使用ライブラリ

- [JijModeling 2.2.0](https://www.documentation.jijzept.com/docs/jijmodeling/) — 数理最適化モデリング
- [OpenJij](https://github.com/OpenJij/OpenJij) — シミュレーテッドアニーリング（ローカル）
- [OMMX](https://github.com/Jij-Inc/ommx) — 最適化モデル交換フォーマット
- [ommx-da4-adapter](https://pypi.org/project/ommx-da4-adapter/) — 富士通デジタルアニーラ接続
- [jijzept-solver](https://pypi.org/project/jijzept-solver/) — JijZept Solver 接続

## ソルバーの使い分け

| ソルバー | 得意な問題 | 選ぶ場面 |
|---------|-----------|---------|
| **富士通デジタルアニーラ** | 大規模QUBO | 離散変数のみ、制約が少ない、大規模 |
| **JijZept Solver** | 汎用数理最適化 | 連続変数混在、不等式制約が多い |
| **OpenJij** | 小規模テスト | 開発・プロトタイピング |
