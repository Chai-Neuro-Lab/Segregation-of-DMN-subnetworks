import os
import numpy as np
import pandas as pd
import nibabel as nib

from scipy.stats import zscore


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_PATH = 'PATH_TO_OUTPUT'
SUBJECTS_CSV = 'PATH_TO_SUBJECTS_CSV'
NETWORK_TABLE = 'PATH_TO_NETWORK_TABLE'

NET_I = 'default_b'
NET_J = 'default_c'
THRESHOLDS = np.linspace(0, 2, 85)

# =============================================================================
# Utility Functions
# =============================================================================

def create_func_gii(data, hemi, map_names):
    """Convert data arrays to functional GIFTI."""
    darrays = []
    for x, map_name in zip(data, map_names):
        darray = nib.gifti.GiftiDataArray(
            np.array(x, dtype='float32'),
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE']
        )
        darray.meta = nib.gifti.GiftiMetaData({'Name': map_name})
        darrays.append(darray)

    meta = nib.gifti.GiftiMetaData({
        'AnatomicalStructurePrimary': 'CortexLeft' if hemi == 'L' else 'CortexRight'
    })
    return nib.GiftiImage(darrays=darrays, meta=meta)


def get_correlation_maps(func, networks, network_labels, hemi):
    """Compute vertex-wise correlation maps with each network's mean BOLD."""
    networks = nib.load(networks).darrays[0].data
    n_networks = len(set(networks))
    n_vertices = len(networks)

    func_gii = nib.load(func)
    time_series = np.vstack([d.data for d in func_gii.darrays])
    time_series = zscore(time_series)

    correlation_maps = []
    for i in range(1, n_networks):
        net_xs = time_series[:, networks == i].mean(axis=1)
        net_corrs = np.array([
            np.corrcoef(net_xs, time_series[:, v])[0, 1]
            for v in range(n_vertices)
        ])
        correlation_maps.append(net_corrs)

    gii = create_func_gii(correlation_maps, hemi=hemi, map_names=network_labels)
    gii_normed = create_func_gii(
        [zscore(m, nan_policy='omit') for m in correlation_maps],
        hemi=hemi, map_names=network_labels
    )
    return gii, gii_normed


def get_overlap(surf_area, correlation_maps, network_labels, net_i, net_j, thresholds):
    """Compute overlap and surface area for two networks at multiple thresholds."""
    surf_area = nib.load(surf_area).darrays[0].data
    correlation_maps = nib.load(correlation_maps).darrays

    net_i_map = correlation_maps[network_labels.index(net_i)].data
    net_j_map = correlation_maps[network_labels.index(net_j)].data

    overlap, surf_area_i, surf_area_j = [], [], []
    for t in thresholds:
        mask_i, mask_j = net_i_map > t, net_j_map > t
        surf_area_total = np.sum(surf_area[mask_i | mask_j])
        surf_area_overlap = np.sum(surf_area[mask_i & mask_j])

        surf_area_i.append(np.sum(surf_area[mask_i]))
        surf_area_j.append(np.sum(surf_area[mask_j]))
        overlap.append(surf_area_overlap / surf_area_total)

    return overlap, surf_area_i, surf_area_j


def save_results(df, data, filename_prefix, thresholds):
    """Save thresholded surface/overlap data to CSV."""
    df_out = pd.concat(
        [df, pd.DataFrame(data, columns=[f"thr_{t:.2f}" for t in thresholds])],
        axis=1
    )
    out_path = os.path.join(OUTPUT_PATH, f"{filename_prefix}.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main pipeline for overlap computation."""
    df = pd.read_csv(SUBJECTS_CSV)
    subjects = list(df['ID'])
    network_labels = list(pd.read_csv(NETWORK_TABLE)['label'])

    overlaps, surf_areas_i, surf_areas_j = [], [], []

    for subject in subjects:
        print(f"Processing subject {subject}...")

        surf_area_file = f"PATH_TO_SURFACE_AREAS/sub-{subject}_surface_area.L.func.gii"
        corr_map_file = f"PATH_TO_CORR_MAPS/{subject}_correlation_maps_normed.L.func.gii"

        overlap, area_i, area_j = get_overlap(
            surf_area_file, corr_map_file, network_labels,
            NET_I, NET_J, thresholds=THRESHOLDS
        )

        overlaps.append(overlap)
        surf_areas_i.append(area_i)
        surf_areas_j.append(area_j)

    # Convert to arrays and handle NaNs
    overlaps = np.nan_to_num(np.array(overlaps), 0)
    surf_areas_i = np.array(surf_areas_i)
    surf_areas_j = np.array(surf_areas_j)

    # Save results
    save_results(df, overlaps, f"thre_surf_{NET_J}_overlap_with_{NET_I}", THRESHOLDS)
    save_results(df, surf_areas_i, f"thre_surf_{NET_I}_surf_area", THRESHOLDS)
    save_results(df, surf_areas_j, f"thre_surf_{NET_J}_surf_area", THRESHOLDS)


if __name__ == "__main__":
    main()
