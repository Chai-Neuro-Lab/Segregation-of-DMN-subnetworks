#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 23:11:21 2025

@author: yinghe
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

# =========================================================
# Configuration
# =========================================================
THRESHOLD_95_DIR = "/path/to/threshold_95_labels/"
THRESHOLD_99_DIR_LEFT = "/path/to/threshold_99_labels/left/"
THRESHOLD_99_DIR_RIGHT = "/path/to/threshold_99_labels/right/"
OUTPUT_DIR = "/path/to/output_directory/"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# Helper Functions
# =========================================================
def load_subject_labels(subj_id, base_dir_left, base_dir_right):
    """Load left and right hemisphere labels and concatenate them."""
    try:
        left_file = os.path.join(base_dir_left, subj_id, "networks.L.label.gii")
        right_file = os.path.join(base_dir_right, subj_id, "networks.R.label.gii")

        left_data = nib.load(left_file).darrays[0].data
        right_data = nib.load(right_file).darrays[0].data

        return np.concatenate([left_data, right_data])
    except Exception as e:
        print(f"Warning: Could not load labels for subject {subj_id}: {e}")
        return None

def compute_similarity_matrix(subject_ids, threshold_95_dir, all_99_labels):
    """Compute ARI similarity matrix between subjects."""
    n_subjs = len(subject_ids)
    sim_matrix = np.zeros((n_subjs, n_subjs))

    for i, subj_i in enumerate(tqdm(subject_ids, desc="Computing similarity")):
        # Load 95% threshold labels
        labels_95 = load_subject_labels(subj_i, threshold_95_dir, threshold_95_dir)
        if labels_95 is None:
            sim_matrix[i, :] = np.nan
            continue

        for j, subj_j in enumerate(subject_ids):
            labels_99 = all_99_labels.get(subj_j)
            if labels_99 is None:
                sim_matrix[i, j] = np.nan
                continue

            sim_matrix[i, j] = adjusted_rand_score(labels_95, labels_99)

    return sim_matrix

# =========================================================
# Main Execution
# =========================================================
if __name__ == "__main__":
    # List all subject IDs from 95% threshold directory
    subject_ids = [d for d in os.listdir(THRESHOLD_95_DIR) 
                   if os.path.isdir(os.path.join(THRESHOLD_95_DIR, d))]

    print(f"Found {len(subject_ids)} subjects.")

    # Preload all 99% threshold labels
    print("Loading 99% threshold labels...")
    all_99_labels = {}
    for subj_id in tqdm(subject_ids, desc="Loading 99% labels"):
        all_99_labels[subj_id] = load_subject_labels(subj_id, THRESHOLD_99_DIR_LEFT, THRESHOLD_99_DIR_RIGHT)

    # Compute ARI similarity matrix
    similarity_matrix = compute_similarity_matrix(subject_ids, THRESHOLD_95_DIR, all_99_labels)

    # Save results as CSV
    df_sim = pd.DataFrame(similarity_matrix, index=subject_ids, columns=subject_ids)
    output_file = os.path.join(OUTPUT_DIR, "ARI_similarity_matrix.csv")
    df_sim.to_csv(output_file)
    print(f"Similarity matrix saved to: {output_file}")