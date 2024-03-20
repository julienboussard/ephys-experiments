import argparse
import pickle
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from dartsort import TemplateConfig
from dartsort.templates import TemplateData, template_util
from dartsort.util import data_util
from ephysx import ibl_util
from one.api import ONE


def extract_templates(
    pid,
    scratch_dir,
    allsyms_dir,
    data_dir,
    n_jobs=1,
    overwrite=False,
    retry_err=True,
    summarize_errors=False,
    memory_gb=300,
):
    symlink_dir = allsyms_dir / f"syms{pid}"
    temps_dir = data_dir / f"temps{pid}"
    if overwrite and temps_dir.exists():
        shutil.rmtree(symlink_dir)
        shutil.rmtree(temps_dir)

    done = temps_dir.exists() and (temps_dir / "denoised_template_data.npz").exists()
    if done:
        print("already done")
        return

    had_err = temps_dir.exists() and (temps_dir / "error.pkl").exists()
    if had_err:
        with open(temps_dir / "error.pkl", "rb") as jar:
            e = pickle.load(jar)
            print(f"had old err {e=} {str(e)=} {repr(e)=}")
        if retry_err and not summarize_errors:
            (temps_dir / "error.pkl").unlink()
            print("retrying")
        else:
            return
    if summarize_errors:
        return

    temps_dir.mkdir(exist_ok=True)

    # rec0 = ibl_util.read_popeye_cbin_ibl(pid, symlink_dir)
    try:
        one = ONE()
        one
        print("load sorting...")
        sorting, uuids = ibl_util.get_ks_sorting_popeye(pid, one=one, return_uuids=True)

        if symlink_dir.exists():
            shutil.rmtree(symlink_dir)
        rec0 = ibl_util.read_and_lightppx_popeye_cbin_ibl(pid, symlink_dir, one=one)
        traces0 = rec0.get_traces(0, 0, 100)
        if np.isnan(traces0).any():
            raise ValueError(
                f"nans... {traces0.shape=} "
                f"{np.flatnonzero(np.isnan(traces0[:, 0]))=} "
                f"{np.flatnonzero(np.isnan(traces0[0]))=}"
            )
        # rec = rec0
        # rec = rec0.save_to_folder(scratch_dir, n_jobs=n_jobs)
        # 4/2^30=2^28
        rec_mem = np.ceil((rec0.get_num_samples() * rec0.get_num_channels()) / (2**28))
        if rec_mem < memory_gb:
            rec = rec0.save_to_memory(n_jobs=n_jobs)
        else:
            rec = rec0.save_to_folder(scratch_dir, n_jobs=n_jobs)
        # rec.set_times(rec0.get_times())

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
            show_progress=False,
        )
        sorting = data_util.subset_sorting_by_time_samples(
            sorting,
            start_sample=42,
            end_sample=rec.get_num_samples() - 79,
            reference_to_start_sample=False,
        )

        denoised_template_config = TemplateConfig(
            superres_templates=False,
            registered_templates=False,
            realign_peaks=False,
        )
        raw_template_config = TemplateConfig(
            superres_templates=False,
            registered_templates=False,
            realign_peaks=False,
            low_rank_denoising=False,
        )
        print("raw temps...")
        raw_td = TemplateData.from_config(
            rec,
            sorting,
            template_config=raw_template_config,
            # save_folder=temps_dir,
            n_jobs=n_jobs,
            device="cpu",
            # save_npz_name="raw_template_data.npz",
        )
        print("denoised temps...")
        dn_td = TemplateData.from_config(
            rec,
            sorting,
            template_config=denoised_template_config,
            # save_folder=temps_dir,
            n_jobs=n_jobs,
            # save_npz_name="denoised_template_data.npz",
            device="cpu",
        )
        np.savez(
            temps_dir / "templates.npz",
            raw_templates=raw_td.templates,
            denoised_templates=dn_td.templates,
            unit_ids=dn_td.unit_ids,
            spike_counts=dn_td.spike_counts,
            uuids=uuids[dn_td.unit_ids],
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pid")
    ap.add_argument("--data_dir", default="~/ceph/bwm_700")
    ap.add_argument("--allsyms_dir", default="~/ceph/bwm_700_syms")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--retry_err", action="store_true")
    ap.add_argument("--summarize_errors", action="store_true")
    ap.add_argument("--n_jobs", type=int)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    data_dir.mkdir(exist_ok=True)

    allsyms_dir = Path(args.allsyms_dir).expanduser()
    allsyms_dir.mkdir(exist_ok=True)

    log_txt = data_dir / f"log{args.pid}.txt"
    with open(log_txt, "a") as logf:
        sys.stdout = logf
        sys.stderr = logf
        print("bwm_templates", time.strftime("%Y-%m-%d %H:%M"))
        print(f"{sys.executable=}")
        print(f"{args=}")

        with tempfile.TemporaryDirectory() as scratch_dir:
            scratch_dir = Path(scratch_dir)
            extract_templates(
                args.pid,
                scratch_dir / "rectmp",
                allsyms_dir,
                data_dir,
                n_jobs=1,
                overwrite=args.overwrite,
                retry_err=args.retry_err,
                summarize_errors=args.summarize_errors,
            )
