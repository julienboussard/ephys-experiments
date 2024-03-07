# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python [conda env:sp]
#     language: python
#     name: conda-env-sp-py
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from dartsort.templates import TemplateData, template_util
from dartsort import DARTsortSorting, TemplateConfig
from dartsort.util import data_util, waveform_util, hybrid_util
from dartsort.main import subtract
import spikeinterface.full as si
from one.api import ONE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ephysx import ibl_util#, decollider_ibl_tests
import shutil
import pickle
import numpy as np
from dartsort.peel import ThresholdAndFeaturize
import torch
import h5py
import pandas as pd
from dartsort.util import multiprocessing_util
import dartsort
import string
import cloudpickle
from tqdm.auto import trange, tqdm
import traceback

# %%
dartsort.__file__

# %%
denoised_template_config = TemplateConfig(
    superres_templates=False,
    registered_templates=False,
    realign_peaks=False,
)

# %%
raw_template_config = TemplateConfig(
    superres_templates=False,
    registered_templates=False,
    realign_peaks=False,
    low_rank_denoising=False
)

# %%
1

# %%
one = ONE()
one

# %%
df = pd.read_csv("~/ceph/2023_12_bwm_release.csv")
pids = df.pid.values
pids[:5]

# %%
data_dir = Path("~/ceph/bwm_700").expanduser()
data_dir.mkdir(exist_ok=True)

# %%
# shutil.rmtree(data_dir)

# %%
allsyms_dir = Path("~/ceph/bwm_700_syms").expanduser()
allsyms_dir.mkdir(exist_ok=True)

# %%
scratch_dir = Path("/tmp/templatescratchspace").expanduser()
if scratch_dir.exists():
    shutil.rmtree(scratch_dir)

# %% [markdown]
# ## Compute templates and store preprocessed recordings

# %%
n_jobs = 12

# %%
overwrite = False
retry_err = True
summarize_errors = False

for pid in tqdm(pids):
    print(pid)
    symlink_dir = allsyms_dir / f"syms{pid}"
    temps_dir = data_dir / f"temps{pid}"
    sorting_pkl = temps_dir / f"sorting{pid}.pkl"
    if overwrite and temps_dir.exists():
        shutil.rmtree(symlink_dir)
        shutil.rmtree(temps_dir)

    done = temps_dir.exists() and (temps_dir / "denoised_template_data.npz").exists()
    if done:
        print("already done")
        continue

    had_err = temps_dir.exists() and (temps_dir / "error.pkl").exists()
    if had_err:
        with open(temps_dir / "error.pkl", "rb") as jar:
            e = pickle.load(jar)
            print(f"had old err {e=} {str(e)=} {repr(e)=}")
        if retry_err and not summarize_errors:
            (temps_dir / "error.pkl").unlink()
            print("retrying")
        else:
            continue
    if summarize_errors:
        continue

    temps_dir.mkdir(exist_ok=True)

    # rec0 = ibl_util.read_popeye_cbin_ibl(pid, symlink_dir)
    try:
        print("load sorting...")
        if sorting_pkl.exists() and not overwrite and not had_err:
            with open(sorting_pkl, "rb") as jar:
                sorting = pickle.load(jar)
        else:
            sorting = ibl_util.get_ks_sorting_popeye(pid, one=one)

        if symlink_dir.exists():
            shutil.rmtree(symlink_dir)
        rec0 = ibl_util.read_and_lightppx_popeye_cbin_ibl(
            pid, symlink_dir, one=one
        )
        traces0 = rec0.get_traces(0, 0, 100)
        if np.isnan(traces0).any():
            raise ValueError(
                f"nans... {traces0.shape=} "
                f"{np.flatnonzero(np.isnan(traces0[:, 0]))=} "
                f"{np.flatnonzero(np.isnan(traces0[0]))=}"
            )
        # rec = rec0
        # rec = rec0.save_to_folder(scratch_dir, n_jobs=n_jobs)
        if rec0.get_num_samples() // 30_000 < 5000:
            rec = rec0.save_to_memory(n_jobs=n_jobs)
        else:
            rec = rec0.save_to_folder(scratch_dir, n_jobs=0)
        # rec.set_times(rec0.get_times())

        if not sorting_pkl.exists():
            print("preprocess sorting")
            sorting = data_util.subset_sorting_by_time_samples(
                sorting,
                start_sample=42,
                end_sample=rec.get_num_samples() - 79,
                reference_to_start_sample=False,
            )
            sorting = template_util.get_realigned_sorting(
                rec,
                sorting,
                realign_max_sample_shift=60,
                n_jobs=n_jobs,
                device="cpu",
                # show_progress=False,
            )
            sorting = data_util.subset_sorting_by_time_samples(
                sorting,
                start_sample=42,
                end_sample=rec.get_num_samples() - 79,
                reference_to_start_sample=False,
            )
            with open(sorting_pkl, "wb") as jar:
                pickle.dump(sorting, jar)

        print("temps...")
        td = TemplateData.from_config(
            rec,
            sorting,
            template_config=raw_template_config,
            save_folder=temps_dir,
            n_jobs=n_jobs,
            device="cpu",
            save_npz_name="raw_template_data.npz",
        )
        td = TemplateData.from_config(
            rec,
            sorting,
            template_config=denoised_template_config,
            save_folder=temps_dir,
            n_jobs=n_jobs,
            save_npz_name="denoised_template_data.npz",
            device="cpu",
        )
    except Exception as e:
        print(f"{e=} {str(e)=} {repr(e)=}")
        print(traceback.format_exc())
        with open(temps_dir / "error.pkl", "wb") as jar:
            pickle.dump(e, jar)
    else:
        if (temps_dir / "error.pkl").exists():
            print("previously had error but this time survived")
            (temps_dir / "error.pkl").unlink()
    finally:
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir)
    import gc; gc.collect()
    import torch; torch.cuda.empty_cache()

    # subtract(rec, subtraction_folder, n_jobs=4)

# %%
1

# %%
rec

# %%
rec0

# %%
# !ls {scratch_dir}

# %%
