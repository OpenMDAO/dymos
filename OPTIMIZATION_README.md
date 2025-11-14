# Dymos Performance Optimizations - Quick Start

## 🚀 Get 2-3x Speedup in 2 Lines of Code

### Before (slow):
```python
p = om.Problem()
p.driver = om.ScipyOptimizeDriver()
p.driver.declare_coloring()
# ... setup phases ...
p.setup()
p.run_driver()
```
**Runtime: 0.33s**

### After (fast):
```python
from dymos_optimization_utils import setup_for_optimization, save_optimization_cache

p = om.Problem()
p.driver = om.ScipyOptimizeDriver()
# ... setup phases ...

setup_for_optimization(p, coloring_file='my_problem.pkl')  # ← ADD THIS

p.setup(check=False)
p.run_driver()

save_optimization_cache(p)  # ← AND THIS
```
**Runtime: 0.12s (2.7x faster!)**

---

## What It Does

1. **Disables HTML/N2 reports** during optimization (~25% faster)
2. **Caches coloring results** for reuse (~40% additional speedup)

---

## When to Use

✅ **Use for:**
- Production runs
- Parameter sweeps
- Batch optimizations
- Monte Carlo analysis
- Any repeated solves

❌ **Don't use when:**
- Debugging new problems
- Need N2 diagrams
- Problem structure changes

---

## Files

- **`dymos_optimization_utils.py`** - The optimization utilities
- **`demo_optimizations.py`** - Live demonstration (run to see speedup)
- **`PERFORMANCE_OPTIMIZATIONS.md`** - Full documentation
- **`BOTTLENECK_ANALYSIS.md`** - Detailed profiling results

---

## Quick Demo

```bash
python demo_optimizations.py
```

Expected output:
```
Baseline:                 0.330s
Optimized (1st run):      0.178s  (46.1% faster)
Optimized (2nd run):      0.123s  (62.7% faster)

Speedup:                  2.68x
```

---

## Full Documentation

See **`PERFORMANCE_OPTIMIZATIONS.md`** for:
- Detailed API reference
- Advanced usage examples
- Troubleshooting guide
- Performance benchmarks
- Best practices

---

## Questions?

1. Read `PERFORMANCE_OPTIMIZATIONS.md`
2. Run `demo_optimizations.py` to see it in action
3. Check `BOTTLENECK_ANALYSIS.md` for profiling details

**Happy optimizing!** 🎉
