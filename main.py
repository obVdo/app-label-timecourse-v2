"""
app-label-timecourse-v2: Extract label time courses from a SourceEstimate.

Inputs : source_estimate datatype (*-lh.stc + *-rh.stc),
         freesurfer datatype (for atlas labels).
Outputs: timeseries datatype  (timeseries.tsv.gz + timeseries.json),
         surface/data datatype (left.gii + right.gii, one value per vertex per timepoint),
         time course plots per hemisphere,
         HTML report.
"""

import os
import sys
import glob
import gzip
import json as _json
import numpy as np
import nibabel as nib
import nibabel.freesurfer as _fs

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
ensure_output_dirs('out_dir', 'out_surface', 'out_figs', 'out_report')
report_items = []

# == LOAD CONFIG ==
config = load_config()

# == FIND STC FILES ==
# Brainlife may provide separate stc-lh / stc-rh keys or a directory via stc
stc_lh_file = config.get('stc-lh') or ''
stc_input   = config.get('stc') or ''
stc_base    = None

if stc_lh_file and os.path.isfile(stc_lh_file):
    # Direct file paths from Brainlife (stc-lh / stc-rh keys)
    stc_base = stc_lh_file[:-7] if stc_lh_file.endswith('-lh.stc') else stc_lh_file
elif stc_input:
    if os.path.isdir(stc_input):
        lh_files = sorted(glob.glob(os.path.join(stc_input, '*-lh.stc')))
        if lh_files:
            stc_base = lh_files[0][:-7]   # strip '-lh.stc'
    elif os.path.isfile(stc_input + '-lh.stc'):
        stc_base = stc_input

if stc_base is None:
    add_info_to_product(report_items,
                        f"FATAL: No *-lh.stc files found. "
                        f"Checked stc-lh='{stc_lh_file}', stc='{stc_input}'.", "error")
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
atlas      = (config.get('atlas') or 'aparc').strip()
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
label_names  = []
label_tcs    = []
label_verts  = {}   # label_name → (hemi, vertex_indices_in_full_surface)

for label in all_labels:
    try:
        sub_stc = stc.in_label(label)
        tc      = sub_stc.data.mean(axis=0)
        verts   = sub_stc.vertices[0] if label.hemi == 'lh' else sub_stc.vertices[1]
        label_names.append(label.name)
        label_tcs.append(tc)
        label_verts[label.name] = (label.hemi, verts)
    except Exception as e:
        add_info_to_product(report_items, f"Skipped '{label.name}': {e}", "warning")

if not label_tcs:
    add_info_to_product(report_items, "FATAL: No time courses extracted.", "error")
    create_product_json(report_items)
    sys.exit(1)

label_tcs = np.array(label_tcs)   # (n_labels, n_times)
add_info_to_product(report_items,
                    f"Extracted {len(label_names)} label time courses.", "info")

# == SAVE TIMESERIES DATATYPE ==
tsv_path  = os.path.join('out_dir', 'timeseries.tsv.gz')
json_path = os.path.join('out_dir', 'timeseries.json')

with gzip.open(tsv_path, 'wt') as f:
    f.write('\t'.join(label_names) + '\n')
    for row in label_tcs.T:   # rows = timepoints, cols = labels
        f.write('\t'.join(f'{v:.6g}' for v in row) + '\n')

sfreq = float(1.0 / (stc.times[1] - stc.times[0])) if len(stc.times) > 1 else 1.0
with open(json_path, 'w') as f:
    _json.dump({
        'SamplingFrequency': round(sfreq, 4),
        'StartTime':         round(float(stc.times[0]), 6),
        'Columns':           label_names,
        'atlas':             atlas,
        'n_labels':          len(label_names),
    }, f, indent=2)
add_info_to_product(report_items,
                    f"Saved timeseries: {len(label_names)} labels × {len(stc.times)} timepoints",
                    "info")

# == SAVE SURFACE/DATA GIFTI ==
for _hemi, _side in [('lh', 'left'), ('rh', 'right')]:
    if _hemi not in hemis:
        continue
    surf_path = os.path.join(subjects_dir, subject, 'surf', f'{_hemi}.inflated')
    if not os.path.isfile(surf_path):
        add_info_to_product(report_items,
                            f"Skipping GIfTI {_side}: surface not found at {surf_path}", "warning")
        continue
    try:
        coords, _ = _fs.read_geometry(surf_path)
        n_verts   = len(coords)
        vertex_tc = np.zeros((n_verts, len(stc.times)), dtype=np.float32)

        for name, (h, verts) in label_verts.items():
            if h != _hemi:
                continue
            tc_idx = label_names.index(name)
            vertex_tc[verts, :] = label_tcs[tc_idx]

        gii = nib.gifti.GiftiImage()
        for t_idx in range(len(stc.times)):
            da = nib.gifti.GiftiDataArray(
                data=vertex_tc[:, t_idx],
                intent=nib.nifti1.intent_codes['NIFTI_INTENT_TIME_SERIES'],
                datatype='NIFTI_TYPE_FLOAT32',
            )
            gii.add_gifti_data_array(da)

        gii_path = os.path.join('out_surface', f'{_side}.gii')
        nib.save(gii, gii_path)
        add_info_to_product(report_items,
                            f"Saved GIfTI surface ({_side}): {n_verts} vertices × {len(stc.times)} timepoints",
                            "info")
    except Exception as e:
        add_info_to_product(report_items, f"Could not write GIfTI ({_side}): {e}", "warning")

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

# == BRAIN ANNOTATION PLOT ==
try:
    from qtpy.QtWidgets import QApplication
    _qapp = QApplication.instance() or QApplication(sys.argv)

    import pyvista as pv
    pv.OFF_SCREEN = True
    mne.viz.set_3d_backend('pyvistaqt')

    from mne.viz.backends._pyvista import (
        PyVistaFigure, Plotter as PVPlotter, _PyVistaRenderer, _ALL_PLOTTERS,
    )
    import mne.viz.backends.renderer as renderer_mod

    def _patched_build(self):
        if self._plotter is None:
            store_filtered = {k: v for k, v in self.store.items()
                              if k in ('window_size', 'shape', 'border', 'multi_samples')}
            plotter = PVPlotter(off_screen=True, **store_filtered)
            plotter.background_color = self.background_color
            self._plotter = plotter
            try:
                _ALL_PLOTTERS[plotter._id_name] = plotter
            except AttributeError:
                pass
        if self.plotter.iren is not None:
            self.plotter.iren.initialize()
            def safe_update(stime=1, force_redraw=True):
                self.plotter.render()
            self.plotter.update = safe_update
        return self.plotter

    PyVistaFigure._build = _patched_build

    class _OffscreenRenderer(_PyVistaRenderer):
        _kind = 'pyvistaqt'
        def show(self):
            self.figure.plotter.show(auto_close=False)
        def __getattr__(self, name):
            if name.startswith(('_window_', '_dock_', '_enable_', '_disable_')):
                return lambda *a, **kw: None
            raise AttributeError(name)

    renderer_mod.backend._Renderer = _OffscreenRenderer

    Brain = mne.viz.get_brain_class()
    specific_labels = labels_cfg.strip()

    for _hemi in hemis:
        brain = Brain(subject, hemi=_hemi, surf='inflated',
                      subjects_dir=subjects_dir, size=800, background='white')
        brain.add_annotation(atlas, borders=False, alpha=0.7)

        # Highlight selected labels in red if a subset was requested
        if specific_labels:
            for label in all_labels:
                if label.hemi == _hemi:
                    brain.add_label(label, color='red', alpha=0.9, borders=False)

        brain_path = os.path.join('out_figs', f'brain_annotation_{_hemi}.png')
        brain.save_image(brain_path)
        try:
            brain.close()
        except Exception:
            pass

        add_image_to_product(report_items,
                             f'Brain annotation {_hemi.upper()} ({atlas})',
                             filepath=brain_path)
        report.add_image(brain_path,
                         title=f'Brain annotation {_hemi.upper()} ({atlas})')

except Exception as e:
    add_info_to_product(report_items, f"Could not render brain annotation: {e}", "warning")

report.save(os.path.join('out_report', 'report.html'), overwrite=True)

add_info_to_product(report_items,
                    f"Done: {len(label_names)} labels, {len(stc.times)} time points.",
                    "success")
create_product_json(report_items)
print("Done.")
