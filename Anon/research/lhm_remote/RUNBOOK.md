# Runbook

Nine steps. Everything else in this directory is reference; this is the order.

Stop at step 8 and look before integrating anything.

---

**1. Provision a Linux GPU box.**
A10 / A5000 / A6000 / 3090 / 4090 / L40S. 16 GB VRAM minimum, 24 GB comfortable.
Not Blackwell — torch 2.3.0+cu121 has no sm_120 kernels and `setup_lhmpp.sh`
will refuse it. Not an 80 GB card; this is inference.

**2. Copy this directory across.** Only this directory.

    scp -r lhm_remote/ user@box:~/

**3. Build the environment.**

    bash setup_lhmpp.sh          # ~20-40 min, compiles CUDA extensions

**4. Verify before downloading any weights.**

    python verify_environment.py --lhmpp --build

Nonzero exit means stop. Fix the box, or get a different one. Do not skip ahead
to the checkpoint — that is several GB spent on an environment already known to
be broken.

**5. Run the reconstruction.** This fetches the checkpoint itself.

    bash run_lhmpp.sh            # -> outputs/lhmpp_avatar_v01/

Do not download SMPL-X first. Whether `LHMPP-700M-SMPLX-FREE` needs it is the
open question this run answers.

**6. Record what happened.**
Fill in `recorded_after_the_run` in `environment_manifest.json` — GPU, peak
VRAM, wall time, Gaussian count, and whether `human_model_files` was required.

**7. Retrieve the results.**

    scp -r user@box:~/lhm_remote/outputs/ .

Outputs only. Do not redistribute the checkpoints.

**8. Look at `lhmpp_identity_turntable.png`. This is the gate.**

The question is not "is this a good 3D human". It is: *is this recognisably the
same man as `inputs/avatar_identity_camera1.png`, from the side as well as the
front?*

MPFB scored about 2/10. If this is not dramatically better, **stop**. Do not fix
the camera matrices, do not touch the depth compositor, do not adjust the room,
do not build cameras 4–7. None of those are the problem; the reconstruction is.
Report the failure and pick a different reconstruction route.

**9. Only if the turntable passes:** destroy the box, then continue locally with
Gaussian depth adaptation and CAM1 alignment.

---

## If something fails

| symptom | cause |
|---|---|
| `no kernel image is available` | wrong GPU for the torch build — step 1 |
| `pointops` build fails | nvcc missing from PATH: `export PATH=/usr/local/cuda/bin:$PATH` |
| `spconv` import error | CUDA version mismatch — must be the cu121 wheel |
| no `.ply` produced | read `outputs/lhmpp_avatar_v01/reconstruction.log` |
| export wants `human_model_files` | the SMPLX-FREE claim is false; register with MPI and rerun |
