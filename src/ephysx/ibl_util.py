from os import symlink
from pathlib import Path

import spikeinterface.full as si
from brainbox.io.one import SpikeSortingLoader
from dartsort import DARTsortSorting
from ibllib.atlas import AllenAtlas
from one.api import ONE

sdsc_base_path = Path("/mnt/sdceph/users/ibl/data")


def pid2sdscpath(pid):
    assert sdsc_base_path.exists()
    one = ONE()

    eid, probe = one.pid2eid(pid)
    alyx_base_path = one.eid2path(eid)

    rel_path = one.list_datasets(eid, f"raw_ephys_data/{probe}*ap.cbin")
    assert len(rel_path) == 1

    rel_path = Path(rel_path[0])
    searchdir = (
        sdsc_base_path / alyx_base_path.relative_to(one.cache_dir) / rel_path.parent
    )
    pattern = Path(rel_path.name).with_suffix(f".*.cbin")
    glob = list(searchdir.glob(str(pattern)))
    assert len(glob) == 1
    cbin_path = glob[0]
    assert cbin_path.exists()

    # get meta and .ch
    ch_path = list(cbin_path.parent.glob("*ap*.ch"))
    assert len(ch_path) == 1
    ch_path = ch_path[0]
    meta_path = list(cbin_path.parent.glob("*ap*.meta"))
    assert len(meta_path) == 1
    meta_path = meta_path[0]

    return cbin_path, ch_path, meta_path


def read_popeye_cbin_ibl(pid, symlink_folder):
    """Load an IBL cbin as a spikeinterface BaseRecording on Popeye

    Some small files will be stored in symlink_folder, and it's okay for this
    to be on a slow drive, since they're just symlinks.
    """
    cbin_path, ch_path, meta_path = pid2sdscpath(pid)
    symlink_folder = Path(symlink_folder)

    symlink(
        cbin_path,
        symlink_folder / cbin_path.with_suffix("").with_suffix(".ap.cbin").name,
    )
    symlink(
        ch_path,
        symlink_folder / ch_path.with_suffix("").with_suffix(".ap.ch").name,
    )
    symlink(
        meta_path,
        symlink_folder / meta_path.with_suffix("").with_suffix(".ap.meta").name,
    )

    return si.read_cbin_ibl(str(symlink_folder))


def read_and_destripe_popeye_cbin_ibl(
    pid, symlink_folder, num_chunks_per_segment=100, seed=0
):
    rec = read_popeye_cbin_ibl(pid, symlink_folder)
    rec = rec.astype("float32")
    rec = si.highpass_filter(rec)
    bad_channel_ids, channel_labels = si.detect_bad_channels(
        rec, num_random_chunks=num_chunks_per_segment, seed=seed
    )
    rec = si.phase_shift(rec)
    rec = si.interpolate_bad_channels(rec, bad_channel_ids)
    rec = si.highpass_spatial_filter(rec)
    rec = si.zscore(
        rec, mode="mean+std", num_chunks_per_segment=num_chunks_per_segment, seed=seed
    )
    return rec


def get_ks_sorting(pid):
    one = ONE()
    ba = AllenAtlas()
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    times_samples = spikes["times"]
    return DARTsortSorting(times_samples=times_samples, labels=clusters, channels=channels)
