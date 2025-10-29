#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 23:07:47 2025

@author: yinghe
"""
import os
import pandas as pd
import numpy as np
import nibabel as nib

# =========================================================
# Configuration
# =========================================================
DEMO_PATH = "/path/to/subjects_demographics.csv"
DATA_DIR = "/path/to/subject_label_data/"
OUTPUT_DIR = "/path/to/output_directory/"
NETWORKS = [15, 16, 17]  # Target networks
NUM_VERTICES = 32492     # Expected number of vertices per hemisphere

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# Helper Functions
# =========================================================
def load_and_filter_subjects(demo_path):
    """Load demographic file and filter based on scanning time thresholds."""
    df = pd.read_csv(demo_path)
    threshold = np.where(df['Age'] >= 8, 25.5 * (2 / 3), 20.24 * (2 / 3))
    df = df[df['N_minutes'] > threshold].copy()
    return df

def split_age_groups(df):
    """Split the subject DataFrame into age-based subgroups."""
    children = df[df['Age'] <= 12].copy()
    adolescents = df[(df['Age'] > 12) & (df['Age'] < 18)].copy()
    adults = df[df['Age'] >= 18].copy()
    return {'children': children, 'adolescents': adolescents, 'adults': adults}

def process_subject(sub_id, data_dir, networks, num_vertices, group_counts, group_valid):
    """Process one subject’s left and right hemisphere label files."""
    for hemisphere in ['L', 'R']:
        file_path = os.path.join(data_dir, sub_id, f"networks.{hemisphere}.label.gii")
        if not os.path.exists(file_path):
            print(f"Warning: {sub_id} {hemisphere} hemisphere file not found.")
            continue

        try:
            img = nib.load(file_path)
            data = img.darrays[0].data
            if data.size != num_vertices:
                print(f"Warning: {sub_id} {hemisphere} vertex count mismatch ({data.size} vs {num_vertices})")
                continue

            for net in networks:
                group_counts[net][hemisphere] += (data == net).astype(np.int32)
                group_valid[net][hemisphere] += 1

        except Exception as e:
            print(f"Error: Failed to process {sub_id} {hemisphere} hemisphere - {e}")

def save_group_results(group_name, group_counts, group_valid, output_dir, networks, data_type_codes):
    """Compute network proportions and save results as GIFTI files."""
    print(f"\nSaving proportion maps for {group_name} group...")

    for net in networks:
        for hemisphere in ['L', 'R']:
            valid_count = group_valid[net][hemisphere]
            if valid_count == 0:
                print(f"Warning: No valid subjects for {group_name} - Network {net} {hemisphere} hemisphere.")
                continue

            proportions = group_counts[net][hemisphere].astype(np.float32) / valid_count

            # Create GIFTI data structure
            data_array = nib.gifti.GiftiDataArray(proportions)
            data_array.datatype = data_type_codes['NIFTI_TYPE_FLOAT32']

            gii_img = nib.gifti.GiftiImage()
            gii_img.add_gifti_data_array(data_array)

            # Add metadata
            meta = nib.gifti.GiftiMetaData()
            meta.data.append(nib.gifti.GiftiNVPairs("Description", f"Network {net} proportion for {group_name} group"))
            meta.data.append(nib.gifti.GiftiNVPairs("Network", str(net)))
            meta.data.append(nib.gifti.GiftiNVPairs("Hemisphere", hemisphere))
            meta.data.append(nib.gifti.GiftiNVPairs("AgeGroup", group_name))
            meta.data.append(nib.gifti.GiftiNVPairs("ValidSubjects", str(valid_count)))
            gii_img.meta = meta

            # Save file
            filename = os.path.join(output_dir, f"{group_name}_network{net}_proportion.{hemisphere}.func.gii")
            nib.save(gii_img, filename)
            print(f"Saved: {filename} (Valid subjects: {valid_count})")

# =========================================================
# Main Execution
# =========================================================
if __name__ == "__main__":
    print("Loading subject demographics...")
    df = load_and_filter_subjects(DEMO_PATH)
    group_dfs = split_age_groups(df)

    data_type_codes = {
        'NIFTI_TYPE_INT32': 8,
        'NIFTI_TYPE_FLOAT32': 16
    }

    for group_name, group_df in group_dfs.items():
        print(f"\nProcessing {group_name} group ({len(group_df)} subjects)...")
        group_subjects = group_df['ID'].tolist()

        # Initialize data containers
        group_counts = {net: {'L': np.zeros(NUM_VERTICES, dtype=np.int32),
                              'R': np.zeros(NUM_VERTICES, dtype=np.int32)} for net in NETWORKS}
        group_valid = {net: {'L': 0, 'R': 0} for net in NETWORKS}

        # Process each subject
        for i, sub_id in enumerate(group_subjects, 1):
            process_subject(sub_id, DATA_DIR, NETWORKS, NUM_VERTICES, group_counts, group_valid)
            if i % 10 == 0 or i == len(group_subjects):
                print(f"Progress: {i}/{len(group_subjects)} subjects completed.")

        # Save group-level results
        save_group_results(group_name, group_counts, group_valid, OUTPUT_DIR, NETWORKS, data_type_codes)

    print("\nAll groups processed successfully. Results saved to:", OUTPUT_DIR)