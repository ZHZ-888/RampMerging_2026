# Plan: Collect Free Followers Feature Implementation

## Problem Statement

Currently, the platoon formation algorithm addresses:
1. **Random Forest prediction** - Predicts follower states (free-follower vs following-follower)
2. **RL-based platoon splitting** - Splits oversized platoons using a scoring-based RL agent

However, there's an unaddressed situation:
- **Non-oversized platoons with many free followers** - These free followers are not influenced by their leader, reducing platoon efficiency

**Goal**: Enable nearby side-lane AVs to jump in front of free followers to become their new leader, converting free followers into following followers.

## Current Algorithm Context

### Existing Components
- **Scoring-based RL agent** for rating side-lane AVs during oversized platoon splitting
- **Random Forest model** for predicting follower states (both AV and HV)
- **Platoon formation control** for creating and managing platoons
- **Target journal**: Transportation Research Part C (top-tier)

### Key Functions in `platoon_formation2.py`
1. `find_sparse_platoon()` - Identifies platoons with free followers → `dic_sparse_platoon = {leader_id: first_free_follower}`
2. `find_oversizedP_nearbyAV()` - Finds side-lane AVs near oversized platoons
3. Existing RL scoring infrastructure for platoon splitting

## Implementation Strategy

### Approach: Extend Existing Scoring-Based RL Agent

**Rationale**:
- Maintains architectural consistency
- Reuses proven scoring logic
- Requires minimal additional training
- Simplifies maintenance
- Avoids complexity of training a separate agent

### Alternative Considered
- **Separate RL agent** - Rejected due to increased complexity, training overhead, and architectural inconsistency

## Step-by-Step Implementation Plan

### Step 1: Find Candidate Side-Lane AVs for Sparse Platoons

**New Method**: `find_sparseP_nearbyAV()`

**Purpose**: Identify side-lane AVs positioned near free followers in sparse platoons

**Input**:
- `ls_ihB_av`: List of side-lane AVs (descending order by position)
- `dic_sparse_platoon`: Dictionary mapping `{leader_id: first_free_follower}`
- `dic_platoon_members`: Platoon membership information

**Output**:
- `dic_sparse_candidates`: Dictionary mapping `{leader_id: [candidate_av1, candidate_av2, ...]}`

**Logic**:
1. Reverse `ls_ihB_av` to get ascending order (oldest → newest)
2. For each sparse platoon:
   - Get position of first free follower
   - Find side-lane AVs behind this position
   - Collect all viable candidates
3. Return candidate dictionary

**Implementation Notes**:
- Similar structure to `find_oversizedP_nearbyAV()`
- Consider spatial constraints (e.g., minimum/maximum distance)
- Filter out AVs that are already in platoon formation

### Step 2: Extend RL Scoring Function

**Modify**: Existing scoring mechanism to handle multiple scenarios

**New Parameter**: `scenario_type` with values:
- `'split'` - Split oversized platoons (existing)
- `'collect'` - Collect free followers (new)

**Method Signature**:
```python
def score_candidate_av(self, candidate_av, target_id, scenario_type):
    """
    Score a candidate AV for either splitting or collecting
    
    :param candidate_av: ID of candidate side-lane AV
    :param target_id: leader_id (split) or first_free_follower (collect)
    :param scenario_type: 'split' or 'collect'
    :return: float score
    """
```

**Feature Calculation** (common for both scenarios):
- Relative position between candidate AV and target
- Speed difference
- Gap availability on target lane
- Current platoon size
- Distance to merging section
- Traffic density

**RL Agent Integration**:
- Pass `scenario_type` to RL agent state builder
- Agent outputs a score for the candidate AV
- Higher score = better candidate

### Step 3: Implement Collection Execution Logic

**New Method**: `collect_free_followers()`

**Purpose**: Execute lane change for best candidate AV to collect free followers

**Input**:
- `dic_sparse_candidates`: Dictionary of candidates for each sparse platoon
- `dic_sparse_platoon`: Dictionary of sparse platoons
- `dic_platoon_members`: Platoon membership info

**Logic**:
1. Iterate through each sparse platoon with candidates
2. Score all candidate AVs using RL agent
3. Select highest-scoring candidate
4. Execute lane change command
5. Update AV role and tags:
   - Set `dic_tags[best_av] = 1` (mark as leader)
   - Set `dic_AVroleChange[best_av] = 'collect_insert'`
6. Handle exceptions gracefully

**Safety Checks**:
- Verify safe gap for lane change
- Check traffic conditions
- Ensure candidate AV is still available

### Step 4: Integrate into Main Control Flow

**Modify**: Main platoon formation control loop

**Integration Points**:
1. After `find_sparse_platoon()` is called
2. Before or after oversized platoon splitting logic
3. Should not conflict with other lane change operations

**Call Sequence**:
```python
# 1. Identify sparse platoons
dic_sparse_platoon = self.find_sparse_platoon(...)

# 2. Find candidate side-lane AVs
dic_sparse_candidates = self.find_sparseP_nearbyAV(
    ls_ihB_av, dic_sparse_platoon, dic_platoon_members
)

# 3. Execute collection if candidates exist
if dic_sparse_candidates:
    self.collect_free_followers(
        dic_sparse_candidates, dic_sparse_platoon, dic_platoon_members
    )
```

**Coordination with Existing Logic**:
- Ensure one AV doesn't get multiple lane change commands in same timestep
- Priority handling if AV is candidate for both split and collect
- Update tracking structures after successful collection

### Step 5: RL Agent Training Extension

**Training Data Requirements**:
- Label successful/unsuccessful collection attempts
- Reward function considerations:
  - Positive reward for converting free followers
  - Negative reward for unsafe lane changes
  - Bonus for maintaining traffic flow

**State Space Extension**:
- Add scenario type feature
- Include free follower count
- Consider downstream traffic conditions

**Training Approach**:
- Fine-tune existing RL model with new scenario type
- Use transfer learning from split scenario
- Collect training data from simulation runs

### Step 6: Testing and Validation

**Unit Tests**:
- `find_sparseP_nearbyAV()` correctness
- Candidate selection logic
- Lane change safety checks

**Integration Tests**:
- Full simulation runs with various traffic conditions
- Edge cases: no candidates, multiple sparse platoons, conflicts

**Performance Metrics**:
- Free follower reduction rate
- Platoon formation efficiency
- Safety metrics (TTC, conflict rate)
- Computational overhead

**HPC Testing**:
- Run parameter sweep with different AV penetration rates
- Test with varying ramp demands
- Validate across multiple random seeds

## Implementation Order

1. **Phase 1**: Implement `find_sparseP_nearbyAV()` (Step 1)
2. **Phase 2**: Extend scoring mechanism with `scenario_type` (Step 2)
3. **Phase 3**: Implement `collect_free_followers()` (Step 3)
4. **Phase 4**: Integrate into main control flow (Step 4)
5. **Phase 5**: Extend RL training pipeline (Step 5)
6. **Phase 6**: Testing and validation (Step 6)

## Code Files to Modify

### Primary File
- `functions/platoon_formation2.py` - Main implementation

### Supporting Files
- `platoon_split_rl_model/state_builder.py` - Add scenario type to state
- `platoon_split_rl_model/rl_module.py` - Extend training logic
- `platoon_split_rl_model/main_agent_handler.py` - Update RL agent interface

### Testing/Scripts
- Create new test script for validation
- Update HPC slurm scripts if needed
- Add logging for collection events

## Potential Challenges

### Challenge 1: Coordination Complexity
**Issue**: Multiple lane change operations (split, collect, formation) may conflict

**Solution**: 
- Implement priority queue for lane change requests
- Track AVs with pending lane changes
- Clear coordination logic

### Challenge 2: RL Agent Generalization
**Issue**: Agent trained on splitting may not generalize well to collection

**Solution**:
- Careful feature engineering
- Sufficient training data for both scenarios
- Monitor performance metrics separately

### Challenge 3: Computational Overhead
**Issue**: Additional candidate finding and scoring increases computation

**Solution**:
- Efficient spatial indexing for candidate search
- Limit candidates per timestep
- TODO: Add efficiency improvements if burden too high

### Challenge 4: Safety Validation
**Issue**: Lane changes near free followers may create safety risks

**Solution**:
- Conservative gap acceptance criteria
- TTC-based safety checks
- Gradual rollout with extensive testing

## Performance Expectations

### Expected Improvements
- **Platoon formation rate**: +15-25%
- **Free follower reduction**: -40-60%
- **Merging efficiency**: +10-15%

### Computational Cost
- **Additional overhead**: ~5-10% per timestep
- Mitigated by efficient candidate filtering

## Future Enhancements

### Short-term (After Initial Implementation)
- Optimize candidate selection criteria
- Tune RL reward function based on results
- Add adaptive thresholds for sparse platoon detection

### Long-term (Future Research)
- Consider end-to-end RL for entire platoon formation
- Multi-agent coordination (cooperative AVs)
- Dynamic scenario switching based on traffic conditions

## Notes

- Current algorithm aims for top-tier journal (Transportation Research Part C)
- Focus on demonstrating novel contribution: hybrid RF + RL approach for platoon formation
- Collecting free followers addresses a gap in current literature
- Maintain code quality and documentation standards for publication

## References

### Key Functions (Current Implementation)
- `find_sparse_platoon()` - Line ~XXX in platoon_formation2.py
- `find_oversizedP_nearbyAV()` - Line ~XXX in platoon_formation2.py
- `split_oversized_platoon()` - Existing RL-based splitting

### Related Research Areas
- Platoon formation control
- Mixed traffic management (CAV + HV)
- Ramp merging optimization
- Machine learning for traffic control

---

## Summary

This plan extends the existing scoring-based RL agent to handle collection of free followers, maintaining architectural consistency while adding valuable functionality. The implementation follows a phased approach with clear integration points and testing protocols. The feature addresses a meaningful gap in the current algorithm and strengthens the contribution for publication in top-tier journals.
