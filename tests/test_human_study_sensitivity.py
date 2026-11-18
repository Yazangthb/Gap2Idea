import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "dataset"))
from human_study_analyze import exclusion_sensitivity, finite_population_interval


def test_exact_intervals_cover_finite_populations_at_boundaries():
    # Exhaustive exact coverage for every possible composition of a small population.
    population, n, tail = 12, 4, 0.025
    denominator = math.comb(population, n)
    for successes in range(population + 1):
        covered = 0
        for k in range(n + 1):
            if k > successes or n - k > population - successes:
                continue
            probability = math.comb(successes, k) * math.comb(population - successes, n - k) / denominator
            lo, hi = finite_population_interval(k, n, population, tail)
            covered += probability * (lo <= successes <= hi)
        assert covered >= 0.95 - 1e-12


def test_unanimous_sample_has_uncertainty_and_excluded_items_expand_bounds():
    shares = {"A": (0, 4)}
    population = {"A": 12}
    strata = {"A": {"eligible": 10}}
    result = exclusion_sensitivity(shares, population, strata)
    assert result["point_range"] == (0, 2 / 12)
    assert result["interval"][0] == 0
    assert result["interval"][1] > 2 / 12
    assert result["interval"][1] <= 1
