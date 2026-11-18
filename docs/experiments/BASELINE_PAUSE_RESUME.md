# Pause and resume the November baseline campaign

Run these from a machine with Hamza's existing SSH access. Each command pauses only this campaign on that server. To release both servers, run both commands.

H100 (two GPUs):

```bash
ssh hamza@10.100.30.86 'python3 /home/hamza/gap2idea-h100-baselines/campaign_control.py pause'
```

V100 (port 22245, two GPUs):

```bash
ssh -p 22245 hamza@10.90.25.14 'python3 /home/hamza/gap2idea-baselines/campaign_control.py pause'
```

If already logged into a server as an authorized administrator, use `sudo -u hamza python3` followed by that server's script path and `pause`. As Hamza, omit `sudo -u hamza`.

Wait for `"status": "paused"` and the GPU memory report before starting another job. The command blocks new campaign jobs, interrupts its clients, and stops its H100 containers or unloads its V100 model caches. It does not stop unrelated GPU services; an existing approximately 1 GiB viewer on H100 GPU 1 remains running.

Every saved call is durable. The command records saved counts, last unit IDs and file SHA-256 hashes in `checkpoints/paused-<UTC>/CHECKPOINT.json`, with a pointer in `LAST_PAUSE.json`. Predictions, prompts, settings, per-attempt failures, tokens and timings stay in the original `runs/` tree. A partial trailing record is backed up before recovery. An interrupted call that was not saved can repeat on resume; its unreported cost remains missing rather than zero.

After the other work finishes and the required GPUs are available, resume with:

```bash
ssh hamza@10.100.30.86 'python3 /home/hamza/gap2idea-h100-baselines/campaign_control.py resume'
ssh -p 22245 hamza@10.90.25.14 'python3 /home/hamza/gap2idea-baselines/campaign_control.py resume'
```

Resume uses the original frozen plans and skips recorded units, including recorded failed calls. It refuses duplicate campaign clients or required GPUs using over 2 GiB of memory. It does not restart the retired V100 27B queue. H100 order remains GPU 0: Qwen3.5 27B then 4B; GPU 1: Qwen3.8 27B then Qwen3.5 9B. The V100 cohort contains only its existing Qwen3.5 4B/9B jobs.

Use `status` instead of `pause` to inspect saved progress, or `pause --dry-run` to preview the exact process/container scope without stopping anything. Dry-run scope was checked against the live campaign on both hosts. Tail recovery was checked with an isolated valid-prefix/partial-record fixture. Live inference was not interrupted just to exercise the controls.

The deployed helper is archived in [`baseline_revision_h100_2026-11-18/_runtime/`](baseline_revision_h100_2026-11-18/_runtime/). Version 2 corrects host detection and validates `--dry-run` use. This operational amendment changes no scientific settings, model identity, prompt, data, frozen runner source, or main H100 orchestration. The original H100 plan retains the prospective version 1 helper hash; each actual pause records the deployed helper's hash for audit.
