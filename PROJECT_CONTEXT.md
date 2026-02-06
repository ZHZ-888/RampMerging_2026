# Project Context (Updated January 2026)

Purpose
- Simulation and control of AV/HV platooning on an inflow highway using SUMO/TRaCI.
- Hybrid RF + RL approach for platoon formation and merging control.
- Target: Transportation Research Part C publication.

Runtime & dependencies
- Language: Python 3.x
- Package manager: `pip`
- Main libs used: `numpy`, `joblib`, `torch` (RL), TRaCI (SUMO), plus logging/recorder utilities.
- Typical entry: `formation_controller.py` orchestrates platoon formation modules.

Key files / locations (Modular Structure)
- **Platoon Formation (Refactored January 2026)**:
  - `functions/formation_controller.py` — orchestrates all platoon modules
  - `functions/platoon_basic.py` — core operations (tagging, size tracking, speed control)
  - `functions/platoon_oversized_handler.py` — SC1: oversized platoon handling
  - `functions/platoon_sparse_handler.py` — SC2 & SC3: sparse platoon + free follower collection
  - `functions/platoon_lane_manager.py` — lane change behavior management
- **RL Module**:
  - `platoon_split_rl_model/main_agent_handler.py` — RL agent interface
  - `platoon_split_rl_model/rl_module.py` — RL training logic
  - `platoon_split_rl_model/state_builder.py` — feature extraction
- **Models**:
  - `rf_models/follower_state_prediction_model_251121_ndarray.pkl` — RF model for follower state prediction
- **Data Recording**:
  - `functions/data_recording.py` — vehicle states and groups (`dic_vid_groups`, `get_vid_states`)
- **Other**:
  - `requirements.txt` — python dependencies
  - `run.py` — simulation entry script
  - `REFACTORING_SUMMARY.md` — details of January 2026 modular refactoring

Important runtime notes
- Platoon modules expect `traci` and `data_recorder` objects with methods: `get_vid_states`, `dic_vid_groups`, `get_avhid_ptype`, `max_speed`, `length_ih`
- RF model path resolved relative to repo root: `rf_models/follower_state_prediction_model_251121_ndarray.pkl`
- Module dependencies: `PlatoonOversizedHandler` and `PlatoonSparseHandler` require `PlatoonBasic` instance
- TRaCI calls can raise if a vehicle left the network — code defensively checks exceptions

Quick assistant reload tips
- To load platoon formation modules: use `@workspace functions/platoon_*.py`
- To load controller: use `@workspace functions/formation_controller.py`
- To load the whole repo: use `@project`
- Keep `PROJECT_CONTEXT.md` updated with important file paths for faster reloading

Notes for maintainers
- If you move the RF model, update path in `functions/platoon_sparse_handler.py`
- Keep `data_recorder` interface stable (methods used by platoon modules) to avoid runtime errors
- See `REFACTORING_SUMMARY.md` for details on modular structure and migration from `platoon_formation2.py`

