"""Dispatch pipeline jobs.
TODO: not used as of 20211112
"""
from pathlib import Path

class Disbatch:
    """Create sbatch scripts and dispatch them."""
    @classmethod
    def sbatch(cls, key, value):
        """..."""
        return f"#SBATCH --{key}={value}"

    @classmethod
    def allocate_sbatch(cls, slurmargs):
        """..."""
        return '\n'.join([cls.sbatch(*keyval)
                         for keyval in (("node", slurmargs.get("node", 1)),
                                        ("time", slurmargs.get("time", "24:00:00")),
                                        ("exclusive", None),
                                        ("constraint", slurmargs.get("constraint", "cpu")),
                                        ("mem", slurmargs.get("mem", 0)),
                                        ("partition", slurmargs.get("partition", "prod")),
                                        ("account", slurmargs.get("account", "proj83")))])


    def __init__(self, **slurmargs):
        """Initialize with slurm arguments.
        """
        preamble = "#!/bin/bash"
        self._slurm = '\n'.join([preamble, self.allocate_sbatch(slurmargs)])

    def stamp_label(self, l, sbatch_io):
        """Prepare a stamp for a label l."""
        assert sbactch_io in ("output", "error")

        destination = "out" if sbatch_io == "output" else "err"
        return self.sbatch(sbatch_io, f"topological-analysis-{l}.{destination}")

    def slurm(self, job):
        """..."""
        return '\n'.join([self._slurm,
                          self.stamp_label(job, "error"), self.stamp_label(job, "output")])

    def dispatch(self, label, runnable, config, *args, working_dir=None, venv=None,
                 **kwargs):
        """..."""
        try:
            working_dir = Path(working_dir)
        except TypeError:
            working_dir = Path("topological-anlaysis")

        if not working_dir.is_absolute():
            working_dir = Path.cwd() / working_dir

        working_dir.mkdir(exist_ok=True, parents=False)

        #add_symlink(working_dir,  )

        slurm = self.slurm(job=label)

        try:
            path_venv = Path(venv)
        except TypeError:
            pass
        else:
            slurm += f"\nsource {path_venv}"

        code = """
        if __name__=="main":

        """

        command = f"sbatch {label}.sbatch {config}"
