#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 22:17:07 2025

@author: yinghe
"""
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from itertools import combinations
import os
# Parse input arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--subject')
parser.add_argument('--hemi', required=True, choices=['L', 'R'], help='Hemisphere (L/R)')
args = parser.parse_args()
subject = str(args.subject)
hemi = str(args.hemi)

project_dir = ''
output_dir = ''

network_df = pd.read_csv(f'/networks.csv')
network_labels = list(network_df.label)


def write_network_FC(networks, FC):

    # Initialie matrix to track within- and between-network FC.
    network_FC = np.zeros((17,17))

    # Extract mean between-network FC.
    for i, j in combinations(range(17), 2):

        connections = FC[networks == i+1,:][:,networks == j+1]
        valid_connections = connections[~np.isinf(connections)]  
        network_FC[i,j] = np.nanmean(valid_connections) if len(valid_connections) > 0 else np.nan

    # Extract mean within-network FC.
    for i in range(17):
        
        FC_i_i = FC[networks == i+1,:][:,networks == i+1]
        upper_tri = FC_i_i[np.triu_indices_from(FC_i_i, k=1)]
        valid_connections = upper_tri[~np.isinf(upper_tri)] 
        network_FC[i,i]  = np.nanmean(valid_connections) if len(valid_connections) > 0 else np.nan

    # Add lower triangle.
    network_FC = np.triu(network_FC) + np.triu(network_FC, 1).T

    # Write to .csv
    pd.DataFrame(
        network_FC,
        columns=network_labels,
        index=network_labels
    ).to_csv(f'{output_dir}/{subject}_{hemi}.csv')


networks = nib.load(f'{project_dir}/results/{subject}/networks.{hemi}.label.gii').darrays[0].data
#networks = nib.load(f'/host/corin/tank/YH/Code/PFM/template/yeo/networks.32k.{hemi}.label.gii').darrays[0].data
vertex_timeseries = nib.load(f'{project_dir}/results/{subject}/surface_dtseries_smoothed.{hemi}.func.gii')
vertex_timeseries = np.array([darray.data for darray in vertex_timeseries.darrays])
FC = np.corrcoef(vertex_timeseries.T)
FC = np.arctanh(FC)
FC[np.isinf(FC)] = np.nan 
write_network_FC(networks, FC)