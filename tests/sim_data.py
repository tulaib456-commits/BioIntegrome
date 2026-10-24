"""
Simulated datasets where the TRUE answer is known, so every analysis can be
checked against the truth (not just 'it did not crash').
"""
import numpy as np
import pandas as pd


def make_rnaseq(n_genes=2000, n_per_group=6, n_de=100, fold=3.0, seed=0, groups=("Ctrl", "Treat")):
    """
    Negative-binomial RNA-seq counts. The first n_de genes are truly
    up-regulated by `fold` in the LAST group. Library depths differ per sample.
    Returns (df, sample_groups dict, de_gene_names, size_factors).
    """
    rng = np.random.default_rng(seed)
    n_groups = len(groups)
    n_samples = n_per_group * n_groups
    base = rng.lognormal(mean=4.0, sigma=1.5, size=n_genes)
    depth = rng.uniform(0.6, 1.6, size=n_samples)
    mu = np.outer(base, depth)
    sample_groups = {}
    names = []
    for g_i, g in enumerate(groups):
        for r in range(n_per_group):
            names.append(f"{g}_{r + 1}")
            sample_groups[names[-1]] = g
    if n_de:
        mu[:n_de, (n_groups - 1) * n_per_group:] *= fold
    dispersion = 0.1
    r_param = 1.0 / dispersion
    p = r_param / (r_param + mu)
    counts = rng.negative_binomial(r_param, p)
    genes = [f"GENE{i:04d}" for i in range(n_genes)]
    df = pd.DataFrame(counts, columns=names)
    df.insert(0, "Gene", genes)
    return df, sample_groups, genes[:n_de], depth


def make_proteomics(n_prot=800, n_per_group=6, n_de=60, shift=1.5, missing_rate=0.1, seed=1):
    """
    Log2-ratio style proteomics (contains NEGATIVE numbers, like CPTAC data).
    First n_de proteins are shifted by `shift` in group B. Low values go missing more often.
    """
    rng = np.random.default_rng(seed)
    n = n_per_group * 2
    base = rng.normal(0, 1.0, size=(n_prot, 1))
    vals = base + rng.normal(0, 0.4, size=(n_prot, n))
    vals[:n_de, n_per_group:] += shift
    mask = rng.random(vals.shape) < (missing_rate * (1 + (vals < -0.5)))
    vals[mask] = np.nan
    names = [f"A_{i + 1}" for i in range(n_per_group)] + [f"B_{i + 1}" for i in range(n_per_group)]
    df = pd.DataFrame(vals, columns=names)
    df.insert(0, "Gene", [f"PROT{i:04d}" for i in range(n_prot)])
    groups = {s: s.split("_")[0] for s in names}
    return df, groups, [f"PROT{i:04d}" for i in range(n_de)]
