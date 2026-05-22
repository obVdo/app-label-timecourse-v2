"""
app-label-timecourse-v2: Extract label time courses from a SourceEstimate.

Inputs : source_estimate datatype (*-lh.stc + *-rh.stc),
         freesurfer datatype (for atlas labels).
Outputs: label_time_courses.csv (time × labels),
         time course plots per hemisphere,
         HTML report.
"""

import os
import sys
import glob
import numpy as np

app_dir    = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(app_dir)
for search_path in [app_dir, parent_dir]:
    if os.path.isdir(os.path.join(search_path, 'brainlife_utils')):
        sys.path.insert(0, search_path)
        break

from brainlife_utils import (
    setup_matplotlib_backend,
    load_config,
    ensure_output_dirs,
    add_info_to_product,
    add_image_to_product,
    create_product_json,
)

setup_matplotlib_backend()
import matplotlib.pyplot as plt
import mne

# == SETUP ==
ensure_output_dirs('out_dir', 'out_figs', 'out_report')
report_items = []

# == LOAD CONFIG ==
config = load_config()

# == FIND STC FILES ==
stc_input = config.get('stc') or ''
stc_base  = None

if stc_input:
    if os.path.isdir(stc_input):
        lh_files = sorted(glob.glob(os.path.join(stc_input, '*-lh.stc')))
        if lh_files:
            stc_base = lh_files[0][:-7]   # strip '-lh.stc'
    elif os.path.isfile(stc_input + '-lh.stc'):
        stc_base = stc_input

if stc_base is None:
    add_info_to_product(report_items,
                        f"FATAL: No *-lh.stc files found at '{stc_input}'.", "error")
    create_product_json(report_items)
    sys.exit(1)

# == LOAD STC ==
try:
    stc = mne.read_source_estimate(stc_base)
    add_info_to_product(
        report_items,
        f"Loaded STC: {stc.data.shape[0]} vertices, {stc.data.shape[1]} time points "
        f"({stc.times[0]*1000:.0f}–{stc.times[-1]*1000:.0f} ms)",
        "info"
    )
except Exception as e:
    add_info_to_product(report_items, f"FATAL: Could not read STC: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

# == FREESURFER DIRECTORY ==
fs_path    = config.get('freesurfer') or config.get('output') or None
subject    = config.get('subject') or None
subjects_dir = None

if fs_path and os.path.isdir(fs_path):
    fs_path = os.path.abspath(fs_path)
    if os.path.isdir(os.path.join(fs_path, 'mri')):
        subjects_dir = os.path.dirname(fs_path)
        subject      = subject or os.path.basename(fs_path)
    else:
        _subdirs = [d for d in os.listdir(fs_path)
                    if os.path.isdir(os.path.join(fs_path, d, 'mri'))]
        if _subdirs:
            subjects_dir = fs_path
            subject      = subject or _subdirs[0]

if subjects_dir is None:
    subjects_dir = str(mne.datasets.fetch_fsaverage(verbose=False))
    subject      = 'fsaverage'
    add_info_to_product(report_items,
                        "No FreeSurfer dir provided — using built-in fsaverage atlas.", "info")

add_info_to_product(report_items, f"Subject: {subject}, subjects_dir: {subjects_dir}", "info")

# == ATLAS & LABELS ==
atlas      = config.get('atlas') or 'aparc'
hemi_cfg   = config.get('hemi') or 'both'
labels_cfg = config.get('labels') or ''

hemis = ['lh', 'rh'] if hemi_cfg == 'both' else [hemi_cfg]

try:
    all_labels = []
    for hemi in hemis:
        all_labels += mne.read_labels_from_annot(
            subject, parc=atlas, hemi=hemi, subjects_dir=subjects_dir, verbose=False)
    add_info_to_product(report_items,
                        f"Loaded {len(all_labels)} labels from atlas '{atlas}'.", "info")
except Exception as e:
    add_info_to_product(report_items, f"FATAL: Could not read atlas '{atlas}': {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

# Filter to requested labels if specified
if labels_cfg.strip():
    requested  = {l.strip() for l in labels_cfg.split(',')}
    all_labels = [l for l in all_labels
                  if l.name in requested
                  or l.name.removesuffix('-lh').removesuffix('-rh') in requested]
    if not all_labels:
        add_info_to_product(report_items,
                            f"FATAL: None of the requested labels found: {sorted(requested)}",
                            "error")
        create_product_json(report_items)
        sys.exit(1)
    add_info_to_product(report_items, f"Filtered to {len(all_labels)} labels.", "info")

# == EXTRACT TIME COURSES ==
label_names = []
label_tcs   = []

for label in all_labels:
    try:
        tc = stc.in_label(label).data.mean(axis=0)
        label_names.append(label.name)
        label_tcs.append(tc)
    except Exception as e:
        add_info_to_product(report_items, f"Skipped '{label.name}': {e}", "warning")

if not label_tcs:
    add_info_to_product(report_items, "FATAL: No time courses extracted.", "error")
    create_product_json(report_items)
    sys.exit(1)

label_tcs = np.array(label_tcs)   # (n_labels, n_times)
add_info_to_product(report_items,
                    f"Extracted {len(label_names)} label time courses.", "info")

# == SAVE CSV ==
csv_path = os.path.join('out_dir', 'label_time_courses.csv')
header   = 'time,' + ','.join(label_names)
np.savetxt(csv_path, np.column_stack([stc.times, label_tcs.T]),
           delimiter=',', header=header, comments='')
add_info_to_product(report_items, f"Saved: {csv_path}", "info")

# == PLOTS & REPORT ==
report = mne.Report(title='Label Time Courses')
_colors = plt.cm.tab20.colors

for hemi in hemis:
    hemi_idx    = [i for i, n in enumerate(label_names) if n.endswith(f'-{hemi}')]
    hemi_labels = [label_names[i] for i in hemi_idx]
    hemi_tcs    = [label_tcs[i]   for i in hemi_idx]
    if not hemi_labels:
        continue

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, tc) in enumerate(zip(hemi_labels, hemi_tcs)):
        ax.plot(stc.times * 1000, tc,
                color=_colors[i % len(_colors)], lw=1.2,
                label=name.removesuffix(f'-{hemi}'))
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Source amplitude')
    ax.set_title(f'Label time courses — {hemi.upper()}  ({atlas})')
    ax.legend(fontsize=6, ncol=4, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join('out_figs', f'label_tc_{hemi}.png')
    fig.savefig(fig_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    add_image_to_product(report_items, f'Label time courses {hemi.upper()}', filepath=fig_path)
    report.add_image(fig_path, title=f'Label time courses {hemi.upper()} ({atlas})')

report.save(os.path.join('out_report', 'report.html'), overwrite=True)

add_info_to_product(report_items,
                    f"Done: {len(label_names)} labels, {len(stc.times)} time points.",
                    "success")
create_product_json(report_items)
print("Done.")
