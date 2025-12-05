# utils/config.py
"""
Configuration constants for causalDA project.

Author: Boi Mai Quach <quachmaiboi.com>
License: GNU General Public License v3.0
"""

import numpy as np

# Node definitions
NODE_LOOKUP = {
    0: "Facebook",
    1: "Google Ads",
    2: "TikTok",
    3: "Youtube",
    4: "Affiliates",
    5: "conversion",
}

# Name of the activity
ACTIVITY_NAME = "impression"

# Target sink node
TARGET_NODE = "conversion"

# Parent influence weight range
INFLUENCE_FROM_PARENTS = (-0.2, 0.5)

# Time series length
TIME_PERIODS = 365  # 2 years of daily data

# Base value range for impressions
BASE_RANGE = (1000, 2000)

# Number of seeds to generate
N_SEEDS = 1000

# Probability of random edges between non-target nodes
EDGE_PROB = 0.4

PC_ALPHA = 0.2  # Significance level for PC algorithm
ALPHA_LEVEL = 0.05  # Significance level for edge inclusion
BETA = 0.5  # Beta for F-beta score


def sample_conversion_dict():
    """Randomize channel → conversion weights."""
    return {
        "Facebook": np.random.uniform(0.09, 0.12),
        "Google Ads": np.random.uniform(0.1, 0.15),
        "TikTok": np.random.uniform(0.07, 0.20),
        "Youtube": np.random.uniform(0.08, 0.12),
        "Affiliates": np.random.uniform(0.05, 0.15),
    }
