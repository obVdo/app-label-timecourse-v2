# app-label-timecourse-v2

Brainlife.io app to extract label time courses from a MEG/EEG source estimate (STC).

Given a source estimate and a FreeSurfer parcellation atlas, this app extracts the mean source amplitude time course for each requested label and saves the result as a timeseries datatype and a surface GIFTI file.

## Inputs

| Input | Description |
|-------|-------------|
| `stc` | Source estimate directory (`*-lh.stc` + `*-rh.stc`) |
| `freesurfer` | FreeSurfer subject directory (optional — falls back to fsaverage) |

## Outputs

| Output | Description |
|--------|-------------|
| `out_dir/timeseries.tsv.gz` | Label time courses (rows = timepoints, columns = labels) |
| `out_dir/timeseries.json` | Metadata (sampling frequency, label names, atlas) |
| `out_surface/left.gii` | Surface GIFTI — left hemisphere (vertex × time) |
| `out_surface/right.gii` | Surface GIFTI — right hemisphere (vertex × time) |
| `out_figs/label_tc_lh.png` | Time course plot — left hemisphere |
| `out_figs/label_tc_rh.png` | Time course plot — right hemisphere |
| `out_figs/brain_labels_*.png` | Brain surface with selected labels highlighted |
| `out_report/report.html` | HTML report with all plots |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `atlas` | `aparc` | FreeSurfer parcellation atlas (e.g. `aparc`, `aparc.a2009s`) |
| `labels` | _(all)_ | Comma-separated label names to extract (e.g. `G_oc-temp_lat-fusifor`). Leave empty to extract all labels. |
| `hemi` | `both` | Hemisphere: `lh`, `rh`, or `both` |
| `subject` | _(auto)_ | FreeSurfer subject name (auto-detected from directory) |

## Container

