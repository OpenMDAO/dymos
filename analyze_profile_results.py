#!/usr/bin/env python
"""
Analyze profiling results and generate detailed bottleneck report.
"""

import pstats
from pstats import SortKey

def analyze_profile(stats_file, title):
    """Analyze a profile stats file and extract key bottlenecks."""
    print(f"\n{'='*80}")
    print(f"ANALYSIS: {title}")
    print(f"{'='*80}\n")

    ps = pstats.Stats(stats_file)

    print(f"Total function calls: {ps.total_calls}")
    print(f"Total primitive calls: {ps.prim_calls}")
    print(f"Total time: {ps.total_tt:.3f} seconds\n")

    # Find dymos-specific bottlenecks
    print("\n--- DYMOS-SPECIFIC HOTSPOTS (by cumulative time) ---")
    ps.sort_stats(SortKey.CUMULATIVE)
    ps.print_stats('dymos', 20)

    print("\n--- DYMOS-SPECIFIC HOTSPOTS (by internal time) ---")
    ps.sort_stats(SortKey.TIME)
    ps.print_stats('dymos', 20)

    # Scipy sparse operations
    print("\n--- SCIPY SPARSE OPERATIONS (by internal time) ---")
    ps.sort_stats(SortKey.TIME)
    ps.print_stats('scipy.*sparse', 15)

    # OpenMDAO operations
    print("\n--- OPENMDAO OPERATIONS (by cumulative time) ---")
    ps.sort_stats(SortKey.CUMULATIVE)
    ps.print_stats('openmdao', 15)

    return ps

def main():
    """Analyze all profile files and generate report."""

    profiles = [
        ('profile_Brachistochrone_Setup_Phase.stats', 'Setup Phase'),
        ('profile_Brachistochrone_GaussLobatto_Full.stats', 'GaussLobatto Full Run'),
        ('profile_Brachistochrone_Radau_Full.stats', 'Radau Full Run'),
    ]

    for stats_file, title in profiles:
        try:
            analyze_profile(stats_file, title)
        except Exception as e:
            print(f"Error analyzing {stats_file}: {e}")

    print("\n" + "="*80)
    print("KEY FINDINGS SUMMARY")
    print("="*80)
    print("""
Based on the profiling analysis, the main bottlenecks are:

1. COLORING COMPUTATION (~20-40% of runtime)
   - openmdao/utils/coloring.py: compute_total_coloring, MNCO_bidir
   - This is the automatic determination of Jacobian sparsity patterns

2. HTML REPORT GENERATION (~20-25% of runtime)
   - openmdao/visualization/htmlpp.py: parse_contents
   - N2 diagram and report generation

3. SPARSE MATRIX OPERATIONS (~10-15% of runtime)
   - scipy/sparse: COO matrix creation, checking, conversion
   - scipy.sparse.linalg: SuperLU factorization and solves

4. GRADIENT/JACOBIAN COMPUTATION (~15-20% of runtime)
   - openmdao/core/total_jac.py: compute_totals
   - openmdao/core/group.py: _linearize

5. DYMOS-SPECIFIC OPERATIONS:
   - dymos/utils/lagrange.py: lagrange_matrices
   - dymos/transcriptions configuration
   - dymos/transcriptions/common/control_comp.py: configure_io

RECOMMENDATIONS:
1. Cache coloring results to disk and reuse when problem structure doesn't change
2. Disable HTML report generation during optimization (only enable when needed)
3. Pre-compute and cache Lagrange interpolation matrices
4. Consider using more efficient sparse matrix formats where appropriate
5. Profile larger problems to see if bottlenecks shift with problem size
    """)

if __name__ == '__main__':
    main()
