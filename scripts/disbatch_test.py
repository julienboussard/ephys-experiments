import argparse
import sys
from pathlib import Path

from disbatchc import disBatch

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--n_jobs", type=int, default=0)
    ap.add_argument("--env", type=str, required=True)

    if "--" in sys.argv:
        split_ix = sys.argv.index("--")
        args = sys.argv[1:split_ix]
        db_args = sys.argv[split_ix + 1 :]
    else:
        args = sys.argv[1:]
        db_args = []

    print(f"main: {args=} {db_args=}")

    args = ap.parse_args(args)

    db = disBatch.DisBatcher(tasksname="disbatch_test", args=db_args)
    print(f"main: {db=}")

    # determine jobs: for each binary folder, for each config, do the sorting
    out_dir = Path(args.out_dir)
    task_ix = 0
    tasks = {}
    for i in range(args.n_jobs):
        cmd = f"""python -c 'import sys; print(f'job: {i=} {sys.executable=}')'"""

        tasks[task_ix] = (
            f"{{ source ~/.bashrc ; mamba activate {args.env} ; {cmd} }}"
        )
        print(f"main: {task_ix=} {tasks[task_ix]=}")
        db.submit(tasks[task_ix])
        task_ix += 1

    tid2status = db.syncTasks(tasks)
    for tid in tasks:
        print(
            "main: task %d: %s returned %d, matched: %s"
            % (
                tid,
                repr(tasks[tid]),
                tid2status[tid]["ReturnCode"],
                repr(tasks[tid]) == tid2status[tid]["TaskCmd"],
            )
        )

    db.done()
