# v0.1.0 release verification plan

Prepared as part of v0.1 release candidate review. This is a checklist to
execute before tagging/publishing, not a record of an already-completed
run -- none of these steps have been executed as part of preparing
`release/v0.1.0` itself; the release branch was assembled by reconciling
existing, already-hardware-verified pieces (see
`app/push-to-talk/README.md` "Verified (R9700 / gfx1201)" and
`research/baselines/R9700-Q8-M1/`), not by re-running them.

## Gate

Run in order, on a clean machine or a machine with no prior repo/venv/model
state assumed:

```
1. fresh clone
     git clone https://github.com/boxwrench/Nemotron-VoiceChat-ROCm.git
     cd Nemotron-VoiceChat-ROCm

2. setup from documented instructions
     scripts/setup.sh
   (or the individual steps in docs/INSTALL.md, if isolating a failure)

3. build
     verify scripts/build-rocm.sh actually produced the runtime binary
     at the pinned commit in runtime/README.md -- confirm the pin (A or
     B, see below) matches what was intended for this release before
     starting.

4. convert
     verify scripts/convert-q8.sh produced all four Q8 artifacts and
     their SHA256 hashes match docs/MODELS.md's manifest.

5. smoke test
     scripts/smoke-test.sh
     confirm BUILD / LOAD / STT / TTS / S2S all report pass.

6. PTT --test
     python3 -m venv app/push-to-talk/.venv
     app/push-to-talk/.venv/bin/pip install -r app/push-to-talk/requirements.txt
     app/push-to-talk/.venv/bin/python app/push-to-talk/ptt_terminal.py --test
     confirm: two turns complete against the real build/model, streamed
     text observed, WAV written, process stayed resident across both
     turns (no reload), clean exit.

7. live terminal PTT
     app/push-to-talk/.venv/bin/python app/push-to-talk/ptt_terminal.py
     with a real microphone and speakers/headphones attached.

8. at least two consecutive live turns
     confirm the second turn does not reload the model, and that
     conversational state (multi-turn context) carries across turns.

9. clean quit
     confirm `q` exits the client and the underlying `--serve` process
     both terminate without a hung process or a stack trace.
```

## Required additional case: multi-GPU host with one GPU isolated

Per the documented limitation in `docs/TROUBLESHOOTING.md` ("Multi-GPU
hosts crash on TTS init unless one ROCm device is isolated"), the full
gate above must also be run at least once on a multi-GPU host with
`ROCR_VISIBLE_DEVICES=<n>` explicitly set, to confirm:

- the documented workaround is sufficient (no crash with one device
  isolated)
- the README/TROUBLESHOOTING guidance is accurate and sufficient for a
  new user to follow without additional help

This case was not part of routine single-GPU R9700 development and needs
its own pass before release, even though the underlying PTT
client/server code itself is unchanged between single- and multi-GPU
hosts.

## Explicitly not part of this gate

- No duplex/barge-in verification -- out of scope for v0.1, see
  `docs/M4-DUPLEX-DESIGN.md`.
- No gfx1100 (RX 7900 XT) hardware pass -- tracked separately under M2,
  PENDING per the main README's status table; v0.1 does not claim gfx1100
  support.
- No multi-GPU *performance* claims -- this gate only verifies the
  documented workaround avoids the crash, not that a second GPU is used
  productively (it isn't, in v0.1).

## Status as of release-candidate preparation

Not yet executed. `release/v0.1.0` was assembled from already-verified
pieces (M1 R9700-Q8-M1 baseline, M3 PTT hardware verification recorded in
`app/push-to-talk/README.md`) rather than by re-running this gate fresh;
this gate exists specifically to catch anything the reconciliation itself
(bringing PTT files forward onto current `main`, potentially advancing
the runtime pin) might have disturbed. Run this gate before tagging.
