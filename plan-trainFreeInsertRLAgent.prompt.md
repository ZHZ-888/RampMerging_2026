# Plan: Train Free-Insert RL Agent for Sparse Platoon Collection

## Overview

You currently have a working RL scoring agent (`RLScoringAgent`) for the **split_insert** scenario (oversized platoons). The new agent will learn to select the best side-lane AV candidate to jump in front of free followers in sparse platoons, converting them from free-following to platoon-following state.

## Background

- **Current Status**: You use Random Forest (RF) to predict follower states (free vs. following) and Reinforcement Learning (RL) to handle oversized platoon splitting.
- **Gap**: Non-oversized platoons with many free followers are not efficiently handled. These free followers are not influenced by their leader.
- **Goal**: Train an RL agent to select nearby side-lane AVs to jump in front of free followers, forming new platoons and converting free followers to following followers.

## Architecture

### Two Separate Agents Approach (Recommended)

1. **Split Agent** (existing): Handles oversized platoons (>11 vehicles)
2. **Free-Insert Agent** (new): Handles sparse platoons with free followers

**Rationale for Separation**:
- Different strategic goals: Split reduces size; Free-insert increases density
- Different state spaces: Split focuses on size thresholds; Free-insert focuses on coupling states
- Different timing: Split is urgent (size violation); Free-insert is opportunistic
- Simpler to train and debug with specialized agents

## Implementation Steps

### Step 1: Create New Agent Handler Class

**File**: `platoon_split_rl_model/main_agent_handler.py`

**Task**: Create `FreeInsertAgentHandler` class mirroring `AgentHandler` structure with free-insert-specific logic.

**Key Components**:
- Cooldown control per sparse platoon leader (`self.last_score_step`)
- Candidate evaluation using RL scoring
- Reward calculation based on successful follower collection
- Insert buffer for delayed reward evaluation
- Model checkpointing every 10k steps

**Differences from Split Agent**:
- State builder calls `build_state_free_insert()` instead of `build_state2()`
- Reward focuses on follower coupling success rather than platoon size balance
- Target position is in front of first free follower (not after platoon leader)

### Step 2: Extend StateBuilder with Free-Insert State

**File**: `platoon_split_rl_model/state_builder.py`

**Task**: Add `build_state_free_insert()` method to create normalized state vector for free-insert scenario.

**State Vector Design** (10 dimensions recommended):

1. **Candidate AV features** (2):
   - Normalized speed: `v_av / vmax`
   - Normalized position: `av_pos / max_lane_pos`

2. **Sparse platoon features** (3):
   - Leader speed: `leader_speed / vmax`
   - Number of free followers: `num_free_followers / max_size`
   - Platoon size: `total_size / max_size`

3. **Target insertion features** (3):
   - Distance to first free follower: `tanh((free_fol_pos - av_pos) / max_gap)`
   - Distance to leader: `tanh((leader_pos - av_pos) / max_gap)`
   - Gap to free follower: `gap_to_free_fol / max_gap` (clipped [0,1])

4. **Safety metrics** (2):
   - Front gap (space ahead after insertion): `front_gap / max_gap`
   - Rear gap (space behind after insertion): `rear_gap / max_gap`

**Note**: Use `tanh` normalization for relative distances to preserve directional information and avoid hard clipping.

### Step 3: Design Reward Function

**File**: `platoon_split_rl_model/main_agent_handler.py` (in `FreeInsertAgentHandler`)

**Task**: Implement `evaluate_insertion_reward()` method that evaluates free-insert success.

**Reward Structure**:

**Base Rewards**:
- Successful lane change: +10.0
- Failed lane change: -5.0

**Outcome-Based Rewards** (evaluated when platoon exits control zone):
- Each free follower successfully coupled: +0.5 per follower
- AV becomes recognized leader: +2.0
- No improvement (followers still free): -1.0

**Safety Penalties**:
- Insufficient front gap (<10m): -2.0
- Insufficient rear gap (<10m): -2.0
- Collision or emergency braking: -10.0

**Evaluation Method**:
```
success = check if inserted AV is now in dic_platoon_members as leader
        AND previously free followers are now in following state
        (check dic_id_preState: 0→1 transition)
```

### Step 4: Integrate Agent into Sparse Handler

**File**: `functions/platoon_sparse_handler.py`

**Task**: Replace heuristic in `collect_free_followers()` with RL agent scoring.

**Current Implementation**:
```python
# Simple heuristic: select first candidate
best_av = candidates[0]
```

**New Implementation**:
```python
# RL scoring: evaluate all candidates and select best
selected_av, selected_state, score = self.free_insert_agent.evaluate_candidates(
    step, leader_id, candidates, dic_sparse_platoon, dic_platoon_members
)
if selected_av:
    self.free_insert_agent.execute_insertion(
        step, leader_id, selected_av, selected_state, score
    )
```

**Integration Points**:
- Pass `FreeInsertAgentHandler` instance to `PlatoonSparseHandler.__init__()`
- Call agent's `run_agent_decision()` method from `collect_free_followers()`
- Use agent's `update_reward()` method when platoons exit control zone

### Step 5: Create Training Script

**File**: `scripts/multi_lane/train_free_insert_agent.py` (new file)

**Task**: Create dedicated training script for free-insert agent.

**Training Configuration**:
- **Traffic demands**: Ramp flows 400-1200 veh/h (iterate through values)
- **AV penetration**: 0.1, 0.2, 0.3 (test varying AV availability)
- **Seeds**: 5 random seeds per configuration
- **Training episodes**: 500-1000 episodes minimum
- **Batch size**: 32 transitions
- **Update frequency**: Every 32+ transitions collected
- **Checkpoint interval**: Every 10,000 steps

**Episode Structure**:
```
1. Initialize SUMO simulation with specific demand/seed
2. Run simulation collecting transitions
3. When sparse platoon detected:
   - Agent evaluates candidates and selects best AV
   - Record (state, action) in insert_buffer
4. When platoon exits control zone:
   - Evaluate reward based on follower coupling success
   - Record transition (state, reward)
5. Trigger training when buffer reaches 32+ samples
6. Save model checkpoint periodically
7. End episode when simulation completes
```

**Logging**:
- Track: episode reward, success rate, number of free followers collected
- Save: loss curves, reward history, model checkpoints
- Output: CSV files with per-episode metrics

### Step 6: Add Reward Evaluation and Buffer Management

**File**: `platoon_split_rl_model/main_agent_handler.py` (in `FreeInsertAgentHandler`)

**Task**: Implement reward evaluation when sparse platoon leader exits control zone.

**Buffer Structure**:
```python
self.insert_buffer = [
    {
        'step': insertion_step,
        'leader_id': sparse_platoon_leader,
        'av_id': inserted_av,
        'state': state_vector,
        'platoon_snapshot': platoon_members_at_insertion,
        'free_followers': list_of_free_follower_ids
    },
    ...
]
```

**Reward Evaluation Process**:
```
1. Check if leader_id reached 'ws_1' (exited control zone)
2. Query current platoon structure:
   - Is inserted_av now a leader? (check dic_tags[av_id] == 1)
   - Are previous free_followers now following? (check dic_id_preState)
3. Calculate reward:
   reward = base_reward + (num_coupled_followers * 0.5)
4. Record transition: agent.record_transition(state, reward)
5. Remove from buffer
6. Trigger training if collected >= 32 transitions
```

**Training Trigger**:
- Similar to split agent: collect 32 transitions → train 5 epochs with batch_size=16
- Clear memory buffer after training
- Save model if at checkpoint step

## Further Considerations

### 1. State Dimension Choice
**Question**: Use 8 dimensions (simpler, matches split agent) or 10 dimensions (includes front/rear gap safety metrics)?

**Recommendation**: Start with 10 dimensions for better safety awareness. The additional safety metrics (front/rear gaps) are critical for preventing collisions during lane changes.

**Alternative**: If training is slow or unstable, reduce to 8 by removing gap metrics and relying on SUMO's built-in lane change safety model.

### 2. Reward Shaping Refinement
**Question**: Should reward scale with number of free followers collected (e.g., +0.5 per follower coupled)?

**Recommendation**: Yes, scale reward proportionally. This incentivizes the agent to prioritize sparse platoons with more free followers (higher impact).

**Formula**:
```
reward = base_success_reward + (num_coupled_followers * 0.5) - safety_penalties
```

**Rationale**: Collecting 5 free followers is more valuable than collecting 1, so the reward should reflect this.

### 3. Training Data Collection Mode
**Question**: Reuse existing `run_mpgc_multi_lane_collect_RF_state_data.py` or create separate `train_free_insert_agent.py`?

**Recommendation**: Create separate `train_free_insert_agent.py` script to isolate free-insert agent training from RF data collection.

**Rationale**:
- Different objectives: RF collects follower state labels; RL collects (state, reward) pairs
- Different logging: RL needs loss curves, episode rewards, checkpoint saves
- Cleaner codebase: Separation of concerns

**Reuse**: Both scripts can import same core functions from `formation_controller.py`

### 4. Cooldown Interval Tuning
**Question**: Should `scoring_interval` differ from split agent (currently 10 steps)?

**Recommendation**: Start with 5-step interval for free-insert agent (more frequent opportunities).

**Rationale**:
- Sparse platoons may be less frequent than oversized platoons
- Shorter cooldown increases training data collection rate
- Can adjust based on empirical frequency of sparse platoon detection

**Alternative**: Adaptive cooldown based on detection frequency (implement after baseline works).

### 5. Model Architecture Sharing
**Question**: Reuse `SimpleMLP` from `rl_module.py` or create dedicated network?

**Recommendation**: Reuse `SimpleMLP` architecture but maintain separate model files.

**Implementation**:
- Both agents use `RLScoringAgent` class with same MLP architecture
- Split agent saves to: `saved_models/split_score_model_YYMMDD_HHMM.pt`
- Free-insert agent saves to: `saved_models/free_insert_model_YYMMDD_HHMM.pt`

**Benefits**:
- Code reuse: Both agents use same training loop, loss function, optimizer
- Easy comparison: Same architecture makes performance comparison meaningful
- Separate weights: No risk of interference between two tasks

### 6. Gating Mechanism
**Question**: Should free-insert agent use a gating threshold like split agent?

**Current Split Agent**: Uses `gating_value` to filter low-scoring candidates (only execute if score > threshold).

**Recommendation**: Yes, implement similar gating for free-insert agent.

**Suggested Threshold**: 0.5 (agents learn scores in [0,1] range via Sigmoid output)

**Rationale**: Prevents poor insertions when no good candidates available. Agent can "choose not to act" by scoring all candidates below threshold.

### 7. Multi-Step vs Single-Step Prediction
**Question**: Should agent consider multi-step consequences (future traffic states)?

**Current Approach**: Single-step reward evaluation (check success when platoon exits control zone).

**Recommendation**: Start with single-step (simpler), consider multi-step later if needed.

**Future Enhancement**: Discount future rewards with gamma factor (γ=0.99) to account for long-term platoon stability beyond control zone.

### 8. Curriculum Learning
**Question**: Should training difficulty increase gradually?

**Recommendation**: Yes, implement simple curriculum:

**Stage 1** (Episodes 0-200): Low traffic demand (400-600 veh/h), high AV penetration (0.3)
- Easy scenario: Many AV candidates, less congestion
- Focus: Learn basic insertion mechanics

**Stage 2** (Episodes 200-500): Medium demand (700-900 veh/h), medium penetration (0.2)
- Moderate difficulty: Balanced traffic
- Focus: Learn candidate selection strategy

**Stage 3** (Episodes 500+): High demand (1000-1200 veh/h), low penetration (0.1)
- Hard scenario: Few AVs, dense traffic
- Focus: Learn to exploit rare opportunities

**Implementation**: Adjust traffic parameters in training script based on episode counter.

## Success Metrics

### Training Convergence
- **Loss curve**: Steady decrease over 10k+ steps
- **Episode reward**: Increase from ~0 to positive values
- **Success rate**: >60% of insertions result in follower coupling

### Performance Evaluation
- **Follower collection rate**: Percentage of free followers successfully coupled
- **Safety**: Zero collisions during insertions
- **Efficiency**: Reduce average time free followers spend uncoupled
- **Comparison**: Outperform heuristic baseline (first candidate selection)

### Validation Tests
- Test on unseen traffic demands (e.g., 550, 850 veh/h)
- Test with different seeds than training
- Compare with split agent performance metrics (should be comparable quality)

## Timeline Estimate

1. **Step 1-2** (Code structure): 2-3 hours
2. **Step 3-4** (Reward design & integration): 2-3 hours
3. **Step 5** (Training script): 1-2 hours
4. **Step 6** (Buffer management): 1 hour
5. **Initial training runs**: 4-8 hours (computation time)
6. **Debugging & tuning**: 4-8 hours
7. **Validation & analysis**: 2-4 hours

**Total**: ~2-3 days for complete implementation and initial training results.

## Next Steps

1. Review this plan and confirm approach
2. Decide on state dimension (8 vs 10)
3. Implement `FreeInsertAgentHandler` class
4. Extend `StateBuilder` with new method
5. Create training script
6. Run initial training experiments
7. Analyze results and tune hyperparameters
8. Compare with heuristic baseline
9. Integrate into production pipeline

## Questions to Resolve Before Implementation

1. **State features**: Any additional features needed? (e.g., traffic density, distance to merge point)
2. **Reward weights**: Confirm scaling factor (0.5 per follower) or adjust?
3. **Training environment**: Train on HPC cluster or local machine?
4. **Evaluation protocol**: How to compare free-insert agent vs. split agent fairly?
5. **Integration timing**: When should free-insert agent be called in simulation loop?
