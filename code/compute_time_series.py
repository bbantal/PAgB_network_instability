# -*- coding: utf-8 -*-
"""
This script takes partially preprocessed voxel-space data from fmriprep,
then performs the following procedures:
    -detrends
    -standardizes
    -bandpass filters
    -regresses out confounds
    -parcels into ROIs

Outputs preprocessed ROI-space time-series data.

"""

import os
import sys
import itertools
import pandas as pd
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker
from multiprocessing import Pool

# =============================================================================
# Setup
# =============================================================================

# Define filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../..")) + "/"
WORKDIR = HOMEDIR + "data/fmriprep/"
OUTDIR = HOMEDIR + "data/time_series/"

# Settings
N_NODES = 8  # Number of nodes to be used for computation
CUTOFF = 20  # Trimming time-series

print(f"Number of nodes to be used for computing time-series: {N_NODES}")

# Get run identifiers
subjects = [int(subid) for subid in sys.argv[1:]]  # Subject IDs from bash script
sessions = ["BHB", "GLC"]
tasks = ["rest"]
runs = ["1", "2"]

# Items to be analyzed
items = list(itertools.product(subjects, sessions, tasks, runs))

# Load the parcellation mask
willard_img = image.load_img(HOMEDIR + "utils/499_roi.nii")

# =============================================================================
# Perform computation
# =============================================================================

# Open files and perform analysis
def comp_timeseries(item):
    print("Computing time-series for:", item)

    # Get filepaths
    bold_fp = WORKDIR + ("sub-{0:0>3}/ses-{1}/func/" \
                         "sub-{0:0>3}_ses-{1}_task-{2}_run-{3}_space-MNI152NL"
                         "in2009cAsym_desc-preproc_bold.nii.gz") \
                         .format(item[0], item[1].lower(), item[2], item[3])

    conf_fp = WORKDIR + ("sub-{0:0>3}/ses-{1}/func/" \
                         "sub-{0:0>3}_ses-{1}_task-{2}_run-{3}_desc-confounds_"
                         "regressors.tsv") \
                         .format(item[0], item[1].lower(), item[2], item[3])

    # Load the image and drop first n frames
    func_img = image.index_img(image.load_img(bold_fp), slice(CUTOFF, None))

    # Load confounds
    confounds = pd.read_csv(conf_fp, sep='\t') \
                    .loc[CUTOFF:, [
                               "a_comp_cor_00",
                               "a_comp_cor_01",
                               "a_comp_cor_02",
                               "a_comp_cor_03",
                               "a_comp_cor_04",
                               "a_comp_cor_05",
                               "global_signal",
                               "white_matter",
                               "csf",
                               "trans_x",
                               "trans_y",
                               "trans_z",
                               'rot_x',
                               'rot_y',
                               'rot_z']]

    # Create parcellation object with additional pre-processing parameters
    willard_mask = NiftiLabelsMasker(willard_img, detrend=True,
                                     t_r=0.802, low_pass=0.1, high_pass=0.01,
                                     standardize=True, memory=HOMEDIR+'cache',
                                     memory_level=1)

    # Process and perform parcellation
    roi_time_series = willard_mask.fit_transform(func_img,
                                              confounds=confounds.values)

    # Write into csv
    csv_data = pd.DataFrame(roi_time_series)
    csv_data.to_csv(OUTDIR + "sub-{0:0>3}_ses-{1}_task-{2}_run-" \
                    "{3}.csv".format(item[0], item[1].lower(), item[2], item[3]),
                               header=False, index=False)

# Run computation through multiprocessing
if __name__ == '__main__':
    pool = Pool(processes=N_NODES)
    pool.map(comp_timeseries, items)

