import hashlib
from os import symlink
from pathlib import Path

import numpy as np
import spikeinterface.full as si
from brainbox.io.one import SpikeSortingLoader
from dartsort import DARTsortSorting
from ibllib.atlas import AllenAtlas
from one.api import ONE
import pandas as pd

sdsc_base_path = Path("/mnt/sdceph/users/ibl/data")


def pid2randomseed(pid):
    shake = hashlib.shake_128(pid.encode())
    return int(shake.hexdigest(16), 16)


def pid2sdscpath(pid, one=None):
    assert sdsc_base_path.exists()
    if one is None:
        one = ONE()

    eid, probe = one.pid2eid(pid)
    alyx_base_path = one.eid2path(eid)

    rel_path = one.list_datasets(eid, f"raw_ephys_data/{probe}*ap.cbin")
    if len(rel_path) != 1:
        raise ValueError(f"Wrong number of datasets for ap.cbin: {rel_path}")

    rel_path = Path(rel_path[0])
    searchdir = (
        sdsc_base_path
        / alyx_base_path.relative_to(one.cache_dir)
        / rel_path.parent
    )
    pattern = Path(rel_path.name).with_suffix(f".*.cbin")
    glob = list(searchdir.glob(str(pattern)))
    if len(glob) != 1:
        raise ValueError(f"Wrong number of paths for ap.cbin: {glob}")
    cbin_path = glob[0]
    assert cbin_path.exists()

    # get meta and .ch
    pattern = Path(rel_path.name).with_suffix(f".*.ch")
    ch_path = list(cbin_path.parent.glob(str(pattern)))
    assert len(ch_path) == 1
    ch_path = ch_path[0]
    pattern = Path(rel_path.name).with_suffix(f".*.meta")
    meta_path = list(cbin_path.parent.glob(str(pattern)))
    assert len(meta_path) == 1
    meta_path = meta_path[0]

    return cbin_path, ch_path, meta_path


def read_popeye_cbin_ibl(pid, symlink_folder, one=None):
    """Load an IBL cbin as a spikeinterface BaseRecording on Popeye

    Some small files will be stored in symlink_folder, and it's okay for this
    to be on a slow drive, since they're just symlinks.
    """
    symlink_folder = Path(symlink_folder)
    symlink_folder.mkdir(exist_ok=True)

    # if we already did this, use previously made symlinks
    if list(symlink_folder.glob("*.ap.cbin")):
        return si.read_cbin_ibl(str(symlink_folder))

    cbin_path, ch_path, meta_path = pid2sdscpath(pid, one=one)
    symlink(
        cbin_path,
        symlink_folder
        / cbin_path.with_suffix("")
        .with_suffix("")
        .with_suffix(".ap.cbin")
        .name,
    )
    symlink(
        ch_path,
        symlink_folder
        / ch_path.with_suffix("").with_suffix("").with_suffix(".ap.ch").name,
    )
    symlink(
        meta_path,
        symlink_folder
        / meta_path.with_suffix("")
        .with_suffix("")
        .with_suffix(".ap.meta")
        .name,
    )

    return si.read_cbin_ibl(str(symlink_folder))


def read_and_destripe_popeye_cbin_ibl(
    pid, symlink_folder, one=None, num_chunks_per_segment=100, seed=0
):
    print("read")
    rec = read_popeye_cbin_ibl(pid, symlink_folder, one=one)
    rec = rec.astype("float32")
    rec = si.highpass_filter(rec)
    print("bad chans start...")
    bad_channel_ids, channel_labels = si.detect_bad_channels(
        rec, num_random_chunks=num_chunks_per_segment, seed=seed
    )
    print(f"{bad_channel_ids=}")
    rec = si.phase_shift(rec)
    rec = si.interpolate_bad_channels(rec, bad_channel_ids)
    rec = si.highpass_spatial_filter(rec)
    print("zscore...")
    rec = si.zscore(
        rec,
        mode="mean+std",
        num_chunks_per_segment=num_chunks_per_segment,
        seed=seed,
    )
    return rec


def read_and_lightppx_popeye_cbin_ibl(
    pid, symlink_folder, one=None, num_chunks_per_segment=100, seed=0
):
    rec = read_popeye_cbin_ibl(pid, symlink_folder, one=one)
    rec = rec.astype("float32")
    rec = si.highpass_filter(rec)
    bad_channel_ids, channel_labels = si.detect_bad_channels(
        rec, num_random_chunks=num_chunks_per_segment, seed=seed
    )
    rec = si.phase_shift(rec)
    rec = si.interpolate_bad_channels(rec, bad_channel_ids)
    rec = si.common_reference(rec)
    rec = si.zscore(
        rec,
        mode="mean+std",
        num_chunks_per_segment=num_chunks_per_segment,
        seed=seed,
    )
    return rec


def read_and_catgt_popeye_cbin_ibl(
    pid,
    symlink_folder,
    one=None,
    num_chunks_per_segment=100,
    seed=0,
    remove_bad_channels=True,
):
    rec = read_popeye_cbin_ibl(pid, symlink_folder, one=one)
    rec = rec.astype("float32")
    rec = si.highpass_filter(rec)
    if remove_bad_channels:
        bad_channel_ids, channel_labels = si.detect_bad_channels(
            rec, num_random_chunks=num_chunks_per_segment, seed=seed
        )
        rec = rec.remove_channels(bad_channel_ids)
    rec = si.phase_shift(rec)
    rec = si.common_reference(rec)
    rec = si.zscore(
        rec,
        mode="mean+std",
        num_chunks_per_segment=num_chunks_per_segment,
        seed=seed,
    )
    return rec


def get_ks_sorting(pid, one=None):
    if one is None:
        one = ONE()
    ba = AllenAtlas()
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    print(f'{spikes["times"]=} {spikes["times"].min()=} {spikes["times"].max()=}')
    times_samples = sl.samples2times(spikes["times"], direction="reverse")
    labels = spikes["clusters"].astype(int)
    return DARTsortSorting(
        times_samples=times_samples,
        channels=np.zeros_like(labels),
        labels=labels,
    )


def get_ks_sorting_popeye(pid, one=None, return_uuids=False):
    assert sdsc_base_path.exists()
    if one is None:
        one = ONE()

    eid, probe = one.pid2eid(pid)
    alyx_base_path = one.eid2path(eid)

    rel_path = one.list_datasets(eid, f"alf/{probe}/pykilosort/spikes.samples.npy")
    print(f"{rel_path=}")
    if len(rel_path) != 1:
        print("wrong number of rel path, trying another...")
        rel_path = one.list_datasets(eid, f"alf/{probe}/spikes.samples.npy")
    assert len(rel_path) == 1

    rel_path = Path(rel_path[0])
    searchdir = (
        sdsc_base_path
        / alyx_base_path.relative_to(one.cache_dir)
        / rel_path.parent
    )
    pattern = Path(rel_path.name).with_suffix(f".*.npy")
    glob = list(searchdir.glob(str(pattern)))
    print(f"{len(glob)=}")
    assert len(glob) == 1
    spikes_samples_npy = glob[0]
    assert spikes_samples_npy.exists()
    print(f"{searchdir=} {spikes_samples_npy=}")

    # get labels
    spikes_clusters_npy = searchdir.glob(
        str(
            Path(rel_path.name)
            .with_suffix("")  # npy
            .with_suffix("")  # uuid
            .with_suffix(".clusters.*.npy")
        )
    )
    spikes_clusters_npy = list(spikes_clusters_npy)
    print(f"{len(spikes_clusters_npy)=}")
    assert len(spikes_clusters_npy) == 1
    spikes_clusters_npy = spikes_clusters_npy[0]

    print(f"{pid=} {probe=} {spikes_samples_npy=} {spikes_clusters_npy=}")

    samples = np.load(spikes_samples_npy)
    print(f"{samples.min()=} {samples.max()=}")
    # times_samples = sl.samples2times(times, direction="reverse")
    labels = np.load(spikes_clusters_npy)
    sorting = DARTsortSorting(
        times_samples=samples,
        channels=np.zeros_like(labels),
        labels=labels,
    )
    if not return_uuids:
        return sorting
    
    
    # get uuids
    clusters_uuids_csv = searchdir.glob("clusters.uuids.*.csv")
    clusters_uuids_csv = list(clusters_uuids_csv)
    print(f"{len(clusters_uuids_csv)=}")
    assert len(clusters_uuids_csv) == 1
    clusters_uuids_csv = clusters_uuids_csv[0]
    
    uuids = pd.read_csv(clusters_uuids_csv).uuids.values
    return sorting, uuids
    
