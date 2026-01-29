# Project Context (short)

Purpose
- Simulation and control of AV/HV platooning on an inflow highway using SUMO/TRaCI.

Runtime & dependencies
- Language: Python 3.x
- Package manager: `pip`
- Main libs used: `numpy`, `joblib`, TRaCI (SUMO), plus any logging/recorder utilities.
- Typical entry: a runner script (e.g. `main.py` / `run_simulation.py`) that creates `traci` and `DataRecorder` and instantiates `PlatoonForm`.

Key files / locations
- `functions/platoon_formation2.py` — primary control logic; class `PlatoonForm`.
- `rf_models/follower_state_prediction_model_251121_ndarray.pkl` — RandomForest model loaded by `PlatoonForm.fs_model`.
- `data_recorder.py` (or `utils/data_recorder.py`) — component providing vehicle states and groups referenced by `PlatoonForm` (`dic_vid_groups`, `get_vid_states`, `get_avhid_ptype`, etc.).
- `requirements.txt` — python dependencies (if present).
- `main.py` or `run_simulation.py` — simulation entry script (creates `traci`, `DataRecorder`, runs steps).
- `PROJECT_CONTEXT.md` — this file.

Important runtime notes
- `PlatoonForm` expects a `traci` object and a `data_recorder` object with methods/attributes used in `platoon_formation2.py` (e.g., `get_vid_states`, `dic_vid_groups`, `get_avhid_ptype`).
- Model path is resolved relative to repo root: `rf_models/follower_state_prediction_model_251121_ndarray.pkl`.
- TRaCI calls can raise if a vehicle left the network — code defensively checks exceptions.

Quick assistant reload tips
- To load only this file: use `@workspace functions/platoon_formation2.py`.
- To load the whole repo: use `@project`.
- Keep `PROJECT_CONTEXT.md` updated with any new important file paths or entry points for faster reloading.

Notes for maintainers
- If you move the RF model, update the model path in `functions/platoon_formation2.py` and this file.
- Keep the `data_recorder` interface stable (methods used by `PlatoonForm`) to avoid runtime errors.
