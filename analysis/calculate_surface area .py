#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 23:03:21 2025

@author: yinghe
"""
import os
import numpy as np
import nibabel as nib
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

PROJECT_DIR = "PATH_TO_PROJECT_DIRECTORY"
OUTPUT_DIR = "PATH_TO_OUTPUT_DIRECTORY"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NETWORK_CSV = f"{PROJECT_DIR}/networks.csv"
SUBJECTS_CSV = "PATH_TO_SUBJECTS_CSV"

# Load network names and IDs
NETWORK_NAMES = pd.read_csv(NETWORK_CSV)['label'].tolist()
NETWORK_IDS = np.arange(1, len(NETWORK_NAMES) + 1)  # assumes 1-based indexing in label files

# =============================================================================
# Utility Functions
# =============================================================================

def calculate_network_area(subject_id):
    """
    Compute network-wise surface area percentages for both hemispheres.

    Parameters
    ----------
    subject_id : str
        Subject identifier.

    Returns
    -------
    dict
        Dictionary containing percentage surface area for each network
        in left (L), right (R), and combined hemispheres.
    """
    # Load network label files
    L_label_path = f"{PROJECT_DIR}/results/sub-{subject_id}_networks.L.label.gii"
    R_label_path = f"{PROJECT_DIR}/results/sub-{subject_id}_networks.R.label.gii"

    L_label = nib.load(L_label_path).darrays[0].data
    R_label = nib.load(R_label_path).darrays[0].data

    # Load surface area files
    L_area_path = f"{PROJECT_DIR}/data/sub-{subject_id}/sub-{subject_id}_surface-area.L.func.gii"
    R_area_path = f"{PROJECT_DIR}/data/sub-{subject_id}/sub-{subject_id}_surface-area.R.func.gii"

    L_area = nib.load(L_area_path).darrays[0].data
    R_area = nib.load(R_area_path).darrays[0].data

    # Compute total surface area for each hemisphere
    L_total = np.sum(L_area)
    R_total = np.sum(R_area)

    results = {}

    # Compute % area for each network
    for net_id, name in zip(NETWORK_IDS, NETWORK_NAMES):
        # Left hemisphere
        L_mask = (L_label == net_id)
        L_percent = 100 * np.sum(L_area[L_mask]) / L_total
        results[f"{name}_L"] = L_percent

        # Right hemisphere
        R_mask = (R_label == net_id)
        R_percent = 100 * np.sum(R_area[R_mask]) / R_total
        results[f"{name}_R"] = R_percent

        # Combined
        total_area = np.sum(L_area[L_mask]) + np.sum(R_area[R_mask])
        total_percent = 100 * total_area / (L_total + R_total)
        results[name] = total_percent

    return results


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main entry point for surface area computation."""
    subjects_df = pd.read_csv(SUBJECTS_CSV)
    n_subjects = len(subjects_df)
    print(f"🧠 Starting surface area computation for {n_subjects} subjects...")

    all_results = []
    for i, row in subjects_df.iterrows():
        subject_id = row['ID']
        print(f"[{i+1}/{n_subjects}] Processing subject: {subject_id}")
        try:
            res = calculate_network_area(subject_id)
            all_results.append({**row.to_dict(), **res})
        except Exception as e:
            print(f"⚠️ Failed to process {subject_id}: {e}")

    final_df = pd.DataFrame(all_results)
    output_file = f"{OUTPUT_DIR}/surface_areas_by_network.csv"
    final_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    print(f"Columns included:\n{final_df.columns.tolist()}")


if __name__ == "__main__":
    main()
