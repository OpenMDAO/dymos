# Dymos Performance Bottleneck Analysis

**Date**: 2025-11-13
**Analysis Type**: cProfile-based profiling of optimal control problem solving

## Executive Summary

Profiling of the Dymos codebase identified several key performance bottlenecks. The analysis focused on three scenarios:
1. Setup phase (0.114s)
2. GaussLobatto full optimization (0.516s)
3. Radau full optimization (0.705s)

### Top Bottlenecks by Category

1. **Coloring Computation** (~20-40% of runtime)
2. **HTML Report Generation** (~20-25% of runtime)
3. **Sparse Matrix Operations** (~10-15% of runtime)
4. **Gradient/Jacobian Computation** (~15-20% of runtime)
5. **Lagrange Matrix Computation** (~2-5% of runtime)

---

## Detailed Analysis

### 1. Coloring Computation (CRITICAL BOTTLENECK)

**Impact**: 20-40% of total runtime, particularly severe in Radau transcription (0.264s out of 0.705s)

**Location**:
- `openmdao/utils/coloring.py:compute_total_coloring()`
- `openmdao/utils/coloring.py:MNCO_bidir()`
- `openmdao/utils/coloring.py:_compute_coloring()`

**Description**:
Automatic determination of Jacobian sparsity patterns for efficient derivative computation. This involves:
- Computing full dense Jacobians multiple times (3x)
- Analyzing sparsity patterns
- Determining optimal coloring scheme for parallel derivative computation

**Profiling Data**:
```
GaussLobatto: 0.119s cumulative (23% of total)
Radau:        0.264s cumulative (37% of total)
```

**Recommendations**:
- ✅ **Cache coloring results** to disk and reuse when problem structure unchanged
- Consider user-specified coloring for production runs
- Investigate if coloring can be computed once and reused across similar problems
- For repeated solves, disable dynamic coloring after first run

---

### 2. HTML Report Generation (CRITICAL BOTTLENECK)

**Impact**: 20-25% of total runtime

**Location**:
- `openmdao/visualization/htmlpp.py:parse_contents()` - 0.110-0.114s
- `openmdao/visualization/n2_viewer/n2_viewer.py:_run_n2_report()` - 0.127-0.133s

**Description**:
Generation of interactive N2 diagrams and HTML reports for visualization. This includes:
- Parsing and processing model hierarchy
- Building interactive visualizations
- Writing HTML/JavaScript files

**Profiling Data**:
```
GaussLobatto: 0.114s in htmlpp.parse_contents (22% of total)
              0.133s in N2 report generation (26% of total)
Radau:        0.110s in htmlpp.parse_contents (16% of total)
              0.127s in N2 report generation (18% of total)
```

**Recommendations**:
- ✅ **Disable report generation during optimization** (only enable for debugging)
- Add environment variable or option to suppress visualization hooks
- Consider lazy report generation (only when explicitly requested)
- For batch optimizations, generate reports only for final result

---

### 3. Sparse Matrix Operations (MODERATE BOTTLENECK)

**Impact**: 10-15% of total runtime

**Location**:
- `scipy/sparse/_sputils.py:get_index_dtype()` - Heavy call frequency
- `scipy/sparse/_coo.py:__init__()` and `_check()` - Matrix creation/validation
- `scipy/sparse/_coo.py:_coo_to_compressed()` - Format conversion
- `scipy.sparse.linalg._dsolve._superlu.gstrf()` - LU factorization

**Description**:
Construction, validation, and manipulation of sparse matrices for Jacobians and linear systems.

**Profiling Data**:
```
GaussLobatto:
  - get_index_dtype: 2582 calls, 0.019s
  - SuperLU gstrf: 16 calls, 0.011s
  - COO matrix operations: 0.024s total

Radau:
  - get_index_dtype: 4216 calls, 0.032s
  - SuperLU gstrf: 16 calls, 0.011s
  - Compressed matrix init: 0.147s cumulative
```

**Recommendations**:
- Pre-allocate and reuse sparse matrices where possible
- Investigate more efficient sparse matrix formats (CSR vs COO)
- Consider caching matrix factorizations
- Profile larger problems to see if this scales poorly

---

### 4. Gradient/Jacobian Computation (MODERATE BOTTLENECK)

**Impact**: 15-20% of total runtime

**Location**:
- `openmdao/core/total_jac.py:compute_totals()` - 16 calls, 0.109-0.128s
- `openmdao/core/group.py:_linearize()` - Component linearization
- `openmdao/drivers/scipy_optimizer.py:_gradfunc()` - Gradient function for optimizer

**Description**:
Computing derivatives through the entire model hierarchy. This is inherent to gradient-based optimization.

**Profiling Data**:
```
GaussLobatto: 16 calls to compute_totals, 0.109s total
Radau:        16 calls to compute_totals, 0.128s total
```

**Recommendations**:
- Ensure all components provide analytic derivatives (avoid finite differencing)
- Use sparse Jacobian declarations where applicable
- Consider semi-analytic methods for expensive components
- Already using coloring (good!)

---

### 5. Dymos-Specific Operations

#### 5.1 Lagrange Matrix Computation

**Impact**: 2-5% during setup

**Location**: `dymos/utils/lagrange.py:lagrange_matrices()`

**Description**:
Computation of Lagrange interpolation and differentiation matrices using nested loops and numpy products.

**Profiling Data**:
```
Setup: 60 calls, 0.005s total (4% of setup time)
Runtime: 30-40 calls during optimization
```

**Code Hotspot** (dymos/utils/lagrange.py:4):
```python
# Nested loops with np.prod - O(n^3) complexity
for j in range(nd):
    temp[:] = diff[:]
    temp[:, j] = 1.0
    Li[:, j] = wb[j] * np.prod(temp, axis=1)

for j in range(nd):
    for k in range(nd):
        if k != j:
            temp[:] = diff[:]
            temp[:, j] = 1.0
            temp[:, k] = 1.0
            Di[:, j] += wb[j] * np.prod(temp, axis=1)
```

**Recommendations**:
- ✅ Consider vectorizing the inner loops
- Cache computed matrices (they only depend on grid structure)
- Investigate numba JIT compilation for this function
- Pre-compute common grids at import time

#### 5.2 Control Component Configuration

**Location**: `dymos/transcriptions/common/control_comp.py:configure_io()`

**Impact**: Significant during setup phase (0.014-0.019s)

**Note**: Already uses `@lru_cache` decorator (line 100) which is good practice.

**Profiling Data**:
```
Setup: 0.019s (17% of setup time)
```

#### 5.3 State Interpolation

**Location**:
- `dymos/transcriptions/pseudospectral/components/state_interp_comp.py:compute_partials()`
- Specific methods: `_compute_partials_gauss_lobatto()`, `_compute_partials_radau()`

**Impact**: During optimization iterations

**Profiling Data**:
```
GaussLobatto: 16 calls, 0.015s total
Radau:        16 calls, 0.009s total
```

---

## Performance by Phase

### Setup Phase (0.114s)

| Operation | Time | % of Phase |
|-----------|------|------------|
| OpenMDAO setup/configuration | 0.068s | 60% |
| Dymos phase configuration | 0.053s | 46% |
| Report plugin loading | 0.037s | 32% |
| Sparse matrix creation | 0.019s | 17% |
| Lagrange matrix computation | 0.005s | 4% |

### GaussLobatto Full Run (0.516s)

| Operation | Time | % of Total |
|-----------|------|------------|
| HTML report generation | 0.114s | 22% |
| N2 diagram generation | 0.133s | 26% |
| Coloring computation | 0.119s | 23% |
| Gradient computation | 0.109s | 21% |
| SLSQP optimization | ~0.025s | 5% |
| Other operations | 0.016s | 3% |

**Total overhead from visualization**: 0.247s (48% of total!)

### Radau Full Run (0.705s)

| Operation | Time | % of Total |
|-----------|------|------------|
| Coloring computation | 0.264s | 37% |
| SLSQP optimization | 0.193s | 27% |
| N2 diagram generation | 0.127s | 18% |
| HTML report generation | 0.110s | 16% |
| Gradient computation | 0.128s | 18% |

---

## Optimization Solver Performance

### SLSQP Performance

Both test cases converged in 13 iterations:
- GaussLobatto: ~0.025s pure optimization time
- Radau: ~0.027s pure optimization time

The SLSQP solver itself is NOT a bottleneck. The bottlenecks are in:
1. Supporting infrastructure (reports, coloring)
2. Derivative computation

---

## Recommendations Priority

### HIGH PRIORITY (20%+ impact each)

1. **Disable HTML/N2 report generation during optimization**
   - Add `Problem.setup(reports=False)` or similar option
   - Estimated speedup: 25-45%
   - Implementation: Simple configuration flag

2. **Cache coloring results**
   - Save coloring to disk after first computation
   - Reuse for identical problem structures
   - Estimated speedup: 20-40%
   - Implementation: Medium complexity

### MEDIUM PRIORITY (5-15% impact)

3. **Optimize sparse matrix operations**
   - Pre-allocate matrices where possible
   - Consider different sparse formats
   - Estimated speedup: 5-10%
   - Implementation: Medium complexity

4. **Vectorize Lagrange matrix computation**
   - Remove nested loops in lagrange_matrices()
   - Consider numba compilation
   - Estimated speedup: 2-5%
   - Implementation: Low-medium complexity

### LOW PRIORITY (monitoring)

5. **Profile larger problems**
   - Current analysis on small problems (10 segments)
   - Bottlenecks may shift with problem size
   - Action: Create benchmark suite with varying sizes

6. **Component-level optimization**
   - Most time in infrastructure, not user ODE
   - Focus on Dymos internals rather than example ODEs

---

## Testing Methodology

### Profile Scripts Created

1. `profile_dymos.py` - Main profiling script
   - Tests: Setup, GaussLobatto, Radau transcriptions
   - Generates `.stats` files for detailed analysis

2. `analyze_profile_results.py` - Analysis script
   - Extracts Dymos-specific bottlenecks
   - Categorizes by subsystem (scipy, openmdao, dymos)

### Benchmark Problems

- Problem: Brachistochrone (minimum time)
- Segments: 10-20
- Order: 3
- Variables: 3 states, 1 control, 1 parameter
- Constraints: 2 boundary constraints

---

## Next Steps

1. **Immediate Actions**:
   - Add option to disable reports during optimization
   - Implement coloring cache

2. **Short-term**:
   - Profile with larger problems (50, 100, 200 segments)
   - Benchmark vectorized lagrange_matrices
   - Test impact of each optimization

3. **Long-term**:
   - Continuous performance regression testing
   - Benchmark suite for all transcription methods
   - Document performance best practices for users

---

## Appendix: Profiling Commands

To regenerate this analysis:

```bash
# Run profiling
python profile_dymos.py

# Analyze results
python analyze_profile_results.py

# Interactive exploration
python -m pstats profile_Brachistochrone_GaussLobatto_Full.stats
>>> sort cumtime
>>> stats 30
```

---

## Appendix: System Information

- Python: 3.11
- NumPy: 2.3.4
- SciPy: 1.16.3
- OpenMDAO: 3.41.0
- Dymos: 1.14.1-dev
- Platform: Linux 4.4.0
