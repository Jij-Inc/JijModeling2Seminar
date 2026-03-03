# JijModeling 2 & OMMX Comprehensive Guide

This guide covers the complete workflow for formulating, solving, and extracting results from optimization problems using JijModeling 2 (≥ 2.2.0) and the OMMX ecosystem.

## 1. Critical Syntax Rules

These rules apply everywhere — objectives, constraints, and expressions:

- **`jm.sum()` not `sum()`**: ALL summations MUST use `jm.sum()`, never Python's built-in `sum()`.
- **`&`/`|`/`~` not `and`/`or`/`not`**: Logical operators in comprehension `if` clauses must use bitwise operators, always parenthesized: `if (i > 0) & (i < 10)`.
- **Name elision**: In Decorator API, placeholder and variable names are inferred from the Python variable name. You can omit a name as the first argument. In Plain API, you MUST provide a name explicitly.
- **Decorator replaces function**: `@jm.Problem.define()` replaces the decorated function with a `Problem` object. Use `my_problem.eval(data)` directly — there is no `.problem` attribute.
- **`problem +=` required for constraints**: `problem.Constraint(...)` alone does nothing; you must write `problem += problem.Constraint(...)`.
- **`problem +=` required for objective**: The objective expression must be added via `problem += expr`.
- **Bounds on non-binary vars**: `IntegerVar` and `ContinuousVar` MUST have both `lower_bound` and `upper_bound`. Use `float('inf')` / `-float('inf')` for unbounded.
- **List comprehensions for constraint families**: Use `[expr for i in N]`, not Python for-loops.
- **No `jm.range()`**: `jm.range()` does not exist in JM2 2.2.0. Use `for i in N` (iterating over a Length/Natural placeholder) instead.
- **No `jm.Element()`**: `jm.Element()` is removed. Use generator-expression iteration `for p in P` inside `jm.sum()` and list comprehensions.
- **File execution required**: Decorator API uses `inspect.getsource()` internally, so it MUST be run from a `.py` file (or Jupyter notebook). It will fail with `OSError: could not get source code` when run via `python -c "..."`.

```python
# WRONG — Python for-loop for constraints
for i in range(n):
    problem += problem.Constraint(f"c_{i}", x[i] == 1)

# CORRECT — list comprehension
problem += problem.Constraint("c", [x[i] == 1 for i in N])
```

## 2. Problem Definition

### Decorator API (Recommended)

```python
import jijmodeling as jm

@jm.Problem.define("TSP", sense=jm.ProblemSense.MINIMIZE)
def tsp_problem(problem: jm.DecoratedProblem):
    N = problem.Length(description="Number of cities")
    d = problem.Float(shape=(N, N), description="Distance matrix")
    x = problem.BinaryVar(shape=(N, N), description="Route selection")

    problem += jm.sum(d[i, j] * x[i, j] for i in N for j in N)

    problem += problem.Constraint(
        "visit_once",
        [jm.sum(x[i, j] for j in N) == 1 for i in N]
    )
    problem += problem.Constraint(
        "depart_once",
        [jm.sum(x[i, j] for i in N) == 1 for j in N]
    )

# tsp_problem IS the Problem object (not a function)
instance = tsp_problem.eval(data)  # direct call on the object
```

### Plain API

```python
problem = jm.Problem("Name", sense=jm.ProblemSense.MINIMIZE)
N = problem.Length("N")  # Names are MANDATORY in Plain API
c = problem.Float("c", shape=(N,))
x = problem.BinaryVar("x", shape=(N,))
# ... define objective and constraints ...
```

### Problem Sense

- `jm.ProblemSense.MINIMIZE` — minimize the objective
- `jm.ProblemSense.MAXIMIZE` — maximize the objective (solver adapters handle the sign flip internally)

## 3. Placeholders & Data Patterns

### Placeholder Types

| Method | Alias | Purpose | Example |
|--------|-------|---------|---------|
| `problem.Natural()` | `problem.Length()` | Positive integer scalar (array dimensions) | `N = problem.Natural()` |
| `problem.Float(shape=...)` | — | Real-valued array parameter | `c = problem.Float(shape=(N, M))` |
| `problem.Float()` | — | Real-valued scalar parameter (no shape) | `B = problem.Float(description="Budget")` |
| `problem.Integer(shape=...)` | — | Integer-valued parameter | `caps = problem.Integer(shape=(N,))` |
| `problem.Graph()` | — | Graph structure | `E = problem.Graph()` |
| `problem.CategoryLabel()` | — | Set of categorical identifiers | `edges = problem.CategoryLabel()` |

### Scalar vs Array Parameters

```python
# Scalar — used for single values like counts, limits, budgets
N = problem.Natural(description="Number of items")  # int-like
B = problem.Float(description="Budget limit")        # float-like

# 1D array
v = problem.Float(shape=(N,), description="Item values")

# 2D array
c = problem.Float(shape=(N, M), description="Cost matrix")

# 3D array
d = problem.Float(shape=(P, P, S), description="Pairwise discount per supplier")
```

### CategoryLabel for Sparse/Dictionary Data

```python
edges = problem.CategoryLabel(description="Set of edge identifiers")
weights = problem.Float(shape=edges, description="Edge weights")
flow = problem.ContinuousVar(lower_bound=0, upper_bound=float('inf'), shape=edges)

problem += jm.sum(weights[e] * flow[e] for e in edges)
problem += problem.Constraint("cap", [flow[e] <= weights[e] for e in edges])
```

## 4. Decision Variables

```python
# Binary (0 or 1) — no bounds needed
x = problem.BinaryVar(shape=(N, M), description="Selection")

# Integer — bounds REQUIRED
z = problem.IntegerVar(lower_bound=0, upper_bound=100, shape=(N,), description="Quantities")

# Continuous — bounds REQUIRED; use float('inf') for unbounded
y = problem.ContinuousVar(lower_bound=0.0, upper_bound=float('inf'), shape=(N, M))

# Scalar variable (omit shape)
total = problem.ContinuousVar(lower_bound=0.0, upper_bound=1000.0)
```

### Element-wise Bounds

When different elements need different bounds, overparameterize then constrain:

```python
z = problem.IntegerVar(lower_bound=0, upper_bound=10000, shape=(N,))
K = problem.Float(shape=(N,), description="Per-element upper bounds")
problem += problem.Constraint("z_upper", [z[i] <= K[i] for i in N])
```

## 5. Expressions & Summation

### Basic Patterns

```python
# Single loop
total_value = jm.sum(v[i] * x[i] for i in N)

# Double loop
total_cost = jm.sum(c[p, s] * x[p, s] for p in P for s in S)

# Triple loop
discount = jm.sum(d[p1, p2, s] * x[p1, s] * x[p2, s]
                  for p1 in P for p2 in P for s in S if p1 < p2)

# Quadruple loop (QAP)
qap_obj = jm.sum(f[p1, p2] * d[l1, l2] * x[p1, l1] * x[p2, l2]
                 for p1 in M for p2 in M for l1 in M for l2 in M)
```

### Conditional Summation

```python
# Upper-triangular pairs only (avoid double-counting symmetric data)
jm.sum(Q[i, j] * x[i] * x[j] for i in N for j in N if i < j)

# Triangular range (j from 0 to i-1)
tri = jm.sum(Q[i, j] * x[i] * x[j] for i in N for j in i)

# Parenthesized compound conditions
partial = jm.sum(d[i] * x[i] for i in N if (i > 0) & (i <= 10))

# Excluding diagonal
non_diag = jm.sum(A[i, j] * x[i, j] for i in N for j in M if j != i)
```

### Quadratic Terms (Binary Products)

For binary variables, products like `x[i] * x[j]` or `x[p1, s] * x[p2, s]` are meaningful:
- `x[i] * x[j]` equals 1 only when both `x[i] = 1` and `x[j] = 1`
- This is key for QUBO formulations and interaction terms (discounts, penalties, etc.)

```python
# Discount that applies only when both items are bought from the same supplier
discount = jm.sum(
    d[p1, p2, s] * x[p1, s] * x[p2, s]
    for p1 in P for p2 in P for s in S
    if p1 < p2  # avoid counting each pair twice
)
```

## 6. Objective Function

```python
# Simple linear objective
problem += jm.sum(costs[i, j] * x[i, j] for i in N for j in M)

# Multi-term objective (cost minus discount)
cost = jm.sum(c[p, s] * x[p, s] for p in P for s in S)
discount = jm.sum(
    d[p1, p2, s] * x[p1, s] * x[p2, s]
    for p1 in P for p2 in P for s in S
    if p1 < p2
)
problem += cost - discount

# Quadratic assignment (4-way product)
problem += jm.sum(
    freq[p1, p2] * dist[l1, l2] * x[p1, l1] * x[p2, l2]
    for p1 in M for p2 in M for l1 in M for l2 in M
)
```

**Tip**: You can assign intermediate `jm.sum(...)` expressions to Python variables and combine them with `+`, `-`, `*` before adding to the problem.

## 7. Constraints

### Constraint Families (list comprehension)

```python
# Equality: each part assigned to exactly one supplier
problem += problem.Constraint(
    "one_supplier_per_part",
    [jm.sum(x[p, s] for s in S) == 1 for p in P],
    description="Each part from one supplier",
)

# Inequality: supplier capacity limit
problem += problem.Constraint(
    "supplier_capacity",
    [jm.sum(x[p, s] for p in P) <= K for s in S],
    description="Supplier capacity",
)
```

### Single Constraint (no iteration)

```python
problem += problem.Constraint(
    "budget",
    jm.sum(c[i] * x[i] for i in N) <= B,
    description="Budget constraint",
)
```

### Bidirectional One-Hot (Assignment / Permutation)

When you need a bijection (e.g., QAP: each item to exactly one slot AND each slot to exactly one item):

```python
# Each product to exactly one shelf
problem += problem.Constraint(
    "one_product_per_shelf",
    [jm.sum(x[p, l] for l in M) == 1 for p in M],
)
# Each shelf gets exactly one product
problem += problem.Constraint(
    "one_shelf_per_product",
    [jm.sum(x[p, l] for p in M) == 1 for l in M],
)
```

### Conditional Constraints

```python
problem += problem.Constraint(
    "filtered",
    [jm.sum(x[i, j] for j in M) <= cap[i] for i in N if (i > 0) & (i <= 10)]
)
```

### Choosing the Right Operator

| Semantics | Operator | Examples |
|-----------|----------|----------|
| Exact match | `==` | Demand satisfaction, flow conservation, assignment (one-hot) |
| Cannot exceed | `<=` | Capacity, supply limit, resource budget |
| Must meet minimum | `>=` | Minimum production, coverage requirement |

**Transport/distribution rule**: demand uses `==` (exact), supply uses `<=` (capacity).

### Optional `description` Parameter

All constraints accept an optional `description` keyword argument for documentation:

```python
problem += problem.Constraint(
    "supplier_capacity",
    [jm.sum(x[p, s] for p in P) <= K for s in S],
    description="Each supplier can handle at most K parts",
)
```

## 8. Multi-Period / Time-Indexed Patterns

For problems with time periods `k = 0, 1, ..., K-1`:

- Use `k-1` index arithmetic for linking consecutive periods.
- Split initial period (`k == 0`) from general periods (`k > 0`) when they have different logic.

```python
K = problem.Length(description="Number of periods")

# Initial period (k == 0): uses initial conditions
problem += problem.Constraint(
    "balance_initial",
    [inventory[i, 0] == initial_stock[i] + production[i, 0] - demand_data[i, 0] for i in N]
)

# General periods (k > 0): references k-1
problem += problem.Constraint(
    "balance_general",
    [
        inventory[i, k] == inventory[i, k - 1] + production[i, k] - demand_data[i, k]
        for i in N
        for k in K if k > 0
    ]
)
```

## 9. Advanced Modeling Techniques

### Symmetric Pair Handling

When data is symmetric (e.g., `discount[p1, p2] == discount[p2, p1]`), iterate over unique pairs only using `if p1 < p2`:

```python
# WRONG — counts each pair twice
jm.sum(d[p1, p2] * x[p1] * x[p2] for p1 in N for p2 in N)

# CORRECT — each unordered pair counted once
jm.sum(d[p1, p2] * x[p1] * x[p2] for p1 in N for p2 in N if p1 < p2)
```

In the data dictionary, populate both `d[p1, p2]` and `d[p2, p1]` symmetrically (the `if p1 < p2` condition in the sum handles deduplication).

### Multi-Term Objectives

Break complex objectives into named sub-expressions for readability:

```python
cost = jm.sum(c[p, s] * x[p, s] for p in P for s in S)
discount = jm.sum(
    d[p1, p2, s] * x[p1, s] * x[p2, s]
    for p1 in P for p2 in P for s in S if p1 < p2
)
problem += cost - discount
```

### Quadratic Assignment Problems (QAP)

QAP involves assigning N items to N locations, minimizing pairwise interaction costs:

```python
# 4-way sum: flow[i,j] * distance[k,l] * x[i,k] * x[j,l]
problem += jm.sum(
    f[p1, p2] * d[l1, l2] * x[p1, l1] * x[p2, l2]
    for p1 in M for p2 in M for l1 in M for l2 in M
)
```

QAP always needs bidirectional one-hot constraints (see Section 7).

### Mixing Constraint Types

A single problem can freely mix equality (`==`) and inequality (`<=`, `>=`) constraints:

```python
# Equality: each part from exactly one supplier
problem += problem.Constraint("assign", [jm.sum(x[p, s] for s in S) == 1 for p in P])
# Inequality: supplier capacity
problem += problem.Constraint("cap", [jm.sum(x[p, s] for p in P) <= K for s in S])
```

## 10. OMMX Instance — Bridging Model and Solver

### Creating an Instance

`problem.eval(data_dict)` converts the abstract model into a concrete `ommx.v1.Instance`:

```python
# Keys = placeholder variable names, values = concrete data
data = {
    "N": 4,
    "v": [10, 7, 5, 3],               # 1D list or numpy array
    "c": [[1, 2], [3, 4], [5, 6]],    # 2D nested list or numpy array
    "B": 500.0,                         # scalar
}
instance = my_problem.eval(data)
```

**Rules**:
- Every placeholder declared in the model must have a matching key in the data dictionary.
- The key name must exactly match the Python variable name used in the Decorator API (e.g., `N = problem.Natural()` → key `"N"`).
- Values can be Python lists, numpy arrays, or scalars. Numpy arrays are recommended for large data.
- Shape dimensions must match: if `c` has shape `(P, S)` and `P=30, S=10`, provide a 30×10 array.

### Instance Properties

```python
instance = problem.eval(data)
print(len(instance.decision_variables))  # number of decision variables
```

## 11. Solve Pipeline — OMMX Solver Adapters

JijModeling 2 uses the OMMX ecosystem to connect models to solvers. The OMMX Instance is the universal exchange format — **write the model once, solve with any adapter**.

### OpenJij (Simulated Annealing) — Local, No API Token

```python
from ommx_openjij_adapter import OMMXOpenJijSAAdapter

sample_set = OMMXOpenJijSAAdapter.sample(
    instance,
    num_reads=100,            # number of independent SA runs
    num_sweeps=1000,          # annealing steps per run
    uniform_penalty_weight=50.0,  # penalty coefficient for constraints
)
best = sample_set.best_feasible_unrelaxed
```

### Fujitsu Digital Annealer (DA) — Cloud, Requires Token

```python
from ommx_da4_adapter import OMMXDA4Adapter

sample_set = OMMXDA4Adapter.sample(
    instance,
    token="YOUR_DA_API_TOKEN",
    version="v3c",  # or "v4"
)
best = sample_set.best_feasible_unrelaxed
```

### JijZept Solver — Cloud, Requires Token

```python
import jijzept_solver

# Returns a solution directly (not a SampleSet)
solution = jijzept_solver.solve(
    instance,
    solve_limit_sec=10.0,
)
# solution can be used like `best` from other adapters
```

### `sample()` vs `solve()` — Always Prefer `sample()`

| Method | Returns | On no feasible solution |
|--------|---------|------------------------|
| `OMMXOpenJijSAAdapter.sample()` | `SampleSet` | Returns set; check manually |
| `OMMXOpenJijSAAdapter.solve()` | Best solution | Raises `RuntimeError` |

**Recommendation**: Always use `sample()` + `best_feasible_unrelaxed`. The `solve()` method can crash with `RuntimeError: No feasible solution found` when the relaxed/unrelaxed judgment is strict.

### Solver Interchangeability

The key advantage of OMMX: the same `instance` works with every adapter. To switch solvers, change only the solve call — the model and data stay identical.

```python
instance = problem.eval(data)

# Solver A
result_a = OMMXOpenJijSAAdapter.sample(instance, ...).best_feasible_unrelaxed

# Solver B — same instance, different adapter
result_b = OMMXDA4Adapter.sample(instance, token=TOKEN, ...).best_feasible_unrelaxed
```

## 12. Result Extraction

### Getting the Best Solution

```python
best = sample_set.best_feasible_unrelaxed  # property, not a method call
```

### Decision Variables DataFrame

```python
df = best.decision_variables_df
# Columns: "name", "subscripts", "value", ...
```

### Filtering Selected Variables

```python
# Binary variables: filter for value == 1.0
selected = df[(df["name"] == "x") & (df["value"] == 1.0)]

for _, row in selected.iterrows():
    idx = row["subscripts"][0]         # scalar subscript (1D variable)
    pi, si = row["subscripts"]         # tuple subscript (2D variable)
    val = row["value"]                 # variable value
```

### Objective Value

```python
print(f"Objective: {best.objective}")     # numeric value
print(f"Objective: {best.objective:.1f}") # formatted
```

### Building Solution Structures

```python
# For assignment problems: build a mapping dict
assignments = {}
for _, row in selected.iterrows():
    pi, si = row["subscripts"]
    assignments[pi] = si

# For permutation problems (QAP): build a layout
layout = {}  # shelf_id -> product_id
for _, row in selected.iterrows():
    pi, li = row["subscripts"]
    layout[li] = pi
```

## 13. Penalty Weight Tuning

The `uniform_penalty_weight` parameter controls how strongly constraint violations are penalized in the QUBO formulation.

### Guidelines

| Rule | Detail |
|------|--------|
| **Too small** | Solver finds solutions that violate constraints (infeasible) |
| **Too large** | Objective differences become negligible; solver struggles to distinguish good from bad feasible solutions |
| **Rule of thumb** | Set to **1–10× the typical objective function value** |

### Reference Values from Tested Problems

| Problem Type | Objective Scale | Recommended Penalty |
|-------------|----------------|---------------------|
| Item selection (4 vars) | ~25 | 50 |
| Supplier selection (300 vars) | ~500 | 500–1000 |
| Warehouse layout / QAP (100 vars) | ~8000 | 5000 |

### When Solver Returns No Feasible Solution

1. Increase `uniform_penalty_weight` (try 5–10× current value)
2. Increase `num_reads` (more independent runs → higher chance of finding feasible solutions)
3. Increase `num_sweeps` (more annealing steps per run)
4. Use `sample()` + `best_feasible_unrelaxed` instead of `solve()`

## 14. Common Errors & Gotchas

### `AttributeError: module 'jijmodeling' has no attribute 'range'`

`jm.range()` was removed in JM2 2.2.0. Use `for i in N` iteration instead.

### `OSError: could not get source code`

Decorator API requires source code via `inspect.getsource()`. Run from a `.py` file or Jupyter notebook, not `python -c`.

### `RuntimeError: No feasible solution found`

`solve()` raises this when no feasible solution exists in the sample set. Switch to `sample()` + `best_feasible_unrelaxed` and increase penalty weight.

### `AttributeError: 'Instance' object has no attribute 'decode'`

`instance.decode()` is a JM1 API. In JM2, decoding is handled by the solver adapter. Use `sample_set.best_feasible_unrelaxed`.

### Constraint Has No Effect

`problem.Constraint(...)` returns a constraint object but does not add it. Always use `problem += problem.Constraint(...)`.

### KeyError in `problem.eval(data)`

Every placeholder in the model must have a corresponding key in the data dictionary. Check that key names exactly match the Python variable names.

## 15. Modeling Completeness Checklist

Before finalizing your formulation, verify:

1. **Every given parameter appears in the model**: Each parameter from the problem description must be declared as a placeholder AND used in at least one expression. An unused parameter signals a missing constraint.
2. **Every constraint described in the problem is modeled**: Read the problem statement carefully — each requirement or limitation must appear as a constraint.
3. **Assignment constraints are bidirectional if needed**: If the problem is a permutation (items ↔ slots bijection), you need both row-sum and column-sum one-hot constraints.
4. **Symmetric pairs use `if p1 < p2`**: When summing over unordered pairs, avoid double-counting.
5. **Data dictionary includes all placeholders**: Every placeholder variable name needs a corresponding key with correct dimensions.
6. **Index dimensions match**: If a parameter has shape `(N, K)`, constraints using it must iterate over both dimensions.
7. **Every non-binary variable has explicit bounds**: `IntegerVar` and `ContinuousVar` require both `lower_bound` and `upper_bound`.
8. **Penalty weight is appropriate**: Set `uniform_penalty_weight` to 1–10× the expected objective value scale.
9. **Use `sample()` not `solve()`**: `sample()` + `best_feasible_unrelaxed` is more robust than `solve()`.

## Appendix: Complete End-to-End Example

```python
import jijmodeling as jm
import numpy as np
from ommx_openjij_adapter import OMMXOpenJijSAAdapter

# ---- 1. Define Problem ----
@jm.Problem.define("supplier_selection", sense=jm.ProblemSense.MINIMIZE)
def problem(problem: jm.DecoratedProblem):
    P = problem.Natural(description="Number of parts")
    S = problem.Natural(description="Number of suppliers")
    K = problem.Natural(description="Supplier capacity")
    c = problem.Float(shape=(P, S), description="Cost matrix")
    d = problem.Float(shape=(P, P, S), description="Pair discount")

    x = problem.BinaryVar(shape=(P, S), description="Assignment")

    # Objective: cost - discount
    cost = jm.sum(c[p, s] * x[p, s] for p in P for s in S)
    discount = jm.sum(
        d[p1, p2, s] * x[p1, s] * x[p2, s]
        for p1 in P for p2 in P for s in S if p1 < p2
    )
    problem += cost - discount

    # Constraint: each part from exactly one supplier
    problem += problem.Constraint(
        "one_supplier",
        [jm.sum(x[p, s] for s in S) == 1 for p in P],
    )
    # Constraint: supplier capacity
    problem += problem.Constraint(
        "capacity",
        [jm.sum(x[p, s] for p in P) <= K for s in S],
    )

# ---- 2. Create Instance ----
data = {
    "P": 30, "S": 10, "K": 5,
    "c": cost_matrix,      # numpy array shape (30, 10)
    "d": discount_matrix,  # numpy array shape (30, 30, 10)
}
instance = problem.eval(data)

# ---- 3. Solve ----
sample_set = OMMXOpenJijSAAdapter.sample(
    instance,
    num_reads=500,
    num_sweeps=3000,
    uniform_penalty_weight=1000.0,
)
best = sample_set.best_feasible_unrelaxed

# ---- 4. Extract Results ----
df = best.decision_variables_df
selected = df[(df["name"] == "x") & (df["value"] == 1.0)]
for _, row in selected.iterrows():
    part, supplier = row["subscripts"]
    print(f"Part {part} -> Supplier {supplier}")
print(f"Objective: {best.objective:.1f}")
```
