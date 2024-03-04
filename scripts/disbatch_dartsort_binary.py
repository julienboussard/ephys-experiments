import argparse
import sys
from pathlib import Path

from disbatchc import disBatch

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--names", nargs="+")
    ap.add_argument("--si_rec_paths", nargs="+")
    ap.add_argument("--subtraction_from", nargs="+")
    ap.add_argument("--config_paths", nargs="+")
    ap.add_argument("--out_dir")
    ap.add_argument("--env", default="sp")
    ap.add_argument("--n_jobs_gpu", type=int, default=0)
    ap.add_argument("--n_jobs_cpu", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")

    if "--" in sys.argv:
        split_ix = sys.argv.index("--")
        args = sys.argv[1:split_ix]
        db_args = sys.argv[split_ix + 1 :]
    else:
        args = sys.argv[1:]
        db_args = []

    print(f"{args=} {db_args=}")

    args = ap.parse_args(args)

    db = disBatch.DisBatcher(tasksname="disbatch_dartsort", args=db_args)
    print(f"{db=}")

    assert len(args.si_rec_paths) == len(args.names)
    if args.subtraction_from:
        assert len(args.subtraction_from) == len(args.names)

    # determine jobs: for each binary folder, for each config, do the sorting
    out_dir = Path(args.out_dir)
    task_ix = 0
    tasks = {}
    for config_path in args.config_paths:
        cfg_py = Path(config_path)
        assert cfg_py.exists()

        name = cfg_py.stem
        cfg_out_dir = out_dir / name
        cfg_out_dir.mkdir(exist_ok=True)

        for j, (recname, sipath) in enumerate(
            zip(args.names, args.si_rec_paths)
        ):

            this_out_dir = cfg_out_dir / recname
            this_out_dir.mkdir(exist_ok=True)

            dartsort_cmd = (
                "dartsort_si_config_py "
                f"--config_path {cfg_py} "
                f"--n_jobs_gpu {args.n_jobs_gpu} "
                f"--n_jobs_cpu {args.n_jobs_cpu} "
                f"{sipath} "
                f"{this_out_dir}"
            )
            if args.subtraction_from:
                dartsort_cmd += f" --take_subtraction_from {args.subtraction_from[j]}"
            logfile = this_out_dir / "log.txt"

            tasks[task_ix] = (
                f"{{ source ~/.bashrc ; mamba activate {args.env} ; {dartsort_cmd} ; }} > {logfile}"
            )
            if args.dry_run:
                print(f"{task_ix=} {tasks[task_ix]=}")
            else:
                db.submit(tasks[task_ix])
            task_ix += 1

    tid2status = db.syncTasks(tasks)
    for tid in tasks:
        print(
            "task %d: %s returned %d, matched: %s"
            % (
                tid,
                repr(tasks[tid]),
                tid2status[tid]["ReturnCode"],
                repr(tasks[tid]) == tid2status[tid]["TaskCmd"],
            )
        )

    db.done()
