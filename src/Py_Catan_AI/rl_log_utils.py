import pandas as pd
import numpy as np
from Py_Catan_AI.value_utils import calculate_value_for_first_player
from Py_Catan_AI.rl_decision_model import RLDecisionModel
from Py_Catan_AI.default_structure import default_structure

def finalize_rewards_on_single_game_rl_log(df: pd.DataFrame, gamma=1.0, use_state_value_as_baseline=False) -> pd.DataFrame:
    """
    Compute per-step rewards, discounted returns, and advantages for a SINGLE game's log.
    Also updates 'scaled_heuristic_return' based on heuristic values.
    
    Returns a new DataFrame (don't rely on inplace for masked slices).
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("game_log_df must be a pandas DataFrame")

    # sanity: single game
    uid1 = df["game_UID"].unique() if "game_UID" in df else []
    uid2 = df["game_indicator"].unique() if "game_indicator" in df else []
    uid3 = df["player"].unique() if "player" in df else []
    uid4 = df["tournament_indicator"].unique() if "tournament_indicator" in df else []
    uid5 = df["meta_indicator"].unique() if "meta_indicator" in df else []
    if (len(uid1) != 1 or  len(uid2) != 1 or  len(uid3) != 1 or len(uid4) != 1 or len(uid5) != 1):
        print(uid1, uid2, uid3, uid4, uid5)
        print(df['meta_indicator'] if 'meta_indicator' in df else "no meta_indicator")
        print(df['tournament_indicator'] if 'tournament_indicator' in df else "no tournament_indicator")
        print(df['game_indicator'] if 'game_indicator' in df else "no game_indicator")
        print(df['player'] if 'player' in df else "no player")
        print(df['game_UID'] if 'game_UID' in df else "no game_UID")
        raise ValueError("game_UID, player_name or game_indicator not consistent within game log")

    # Work on a copy to avoid SettingWithCopy issues from masked slices
    df = df.copy()

    # (optional) ensure chronological order; adapt keys if you have a step column
    df = df.sort_values(["round"]).reset_index(drop=False)  # keep old index if needed

    rounds = df["round"].to_numpy()
    scores = df["score"].to_numpy(dtype=float)

    # Specifically for Catan you get the first two points (villages) for free in teh setup phase,
    # irrespective of good or bad decisions, so we remove these from the reward calculation
    scores = [max(0,s-2) for s in scores]

    # Ensure non-decreasing rounds
    if np.any(np.diff(rounds) < 0):
        raise ValueError("round must be non-decreasing within a single game slice")

    # --- Reward = score change per step (first step = 0, not current score) ---
    # By default, np. diff() reduces the array length by one. To retain the original size, 
    # we use append, which adds a 0 at the end
    # so: Scores: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5, 6, 6] 
    # will become: [0 0 0 0 0 0 0 0 1 0 0 0 0 2 0 0 0 2 0 0 0 1 0 0]
    delta_rewards = np.diff(scores, append=scores[-1])

    # --- Compute discounted returns ---
    # we first scale down by their round value, multiply by gamma^round_step
    # then sum future discounted rewards and re-scale by gamma^round_index
    # so effectively for ever step we multiply with 
    # gamma^(round_step - round_index) = gamma^(delta_rounds)
    discounted_delta_reward = delta_rewards * np.power(gamma, rounds)
    returns = np.empty(len(scores), dtype=float)
    for index, round in enumerate(rounds):
        future = sum(discounted_delta_reward[index:]) / np.power(gamma, round)
        returns[index] = future

    # advantage = return - baseline (state_value if present, else 0)
    if "state_value" in df and use_state_value_as_baseline == True: 
        sv = df["state_value"].to_numpy(dtype=float)
    else:
        sv = 0
    advantages = returns - sv

    # assign back positionally
    df.loc[:, "delta_reward"] = delta_rewards
    df.loc[:, "return"] = returns
    df.loc[:, "advantage"] = advantages

    # restore original index if you kept it
    if "index" in df.columns:
        df = df.set_index("index")
        df.index.name = None
        df = df.sort_index()

    # Calculate heuristics as a vectorized array
    df["heuristic"] = df["state"].apply(
        lambda s: calculate_value_for_first_player(default_structure, s)
    ).astype(np.float32)

    # Compute the final value (last heuristic in the series)
    final_val = df["heuristic"].iloc[-1]

    # Compute remaining and scaled heuristic return
    df["scaled_heuristic_return"] = ((final_val - df["heuristic"]) - 0.75) * 2

    # Clean up temporary heuristic column
    df.drop(columns=["heuristic"], inplace=True)

    return df


def split_logs_by_phase(rl_log: pd.DataFrame):
    """
    Split RL log DataFrame into separate logs per phase.
    Returns dict with keys: 'setup', 'gameplay', 'trade'.
    """
    logs = {}
    if rl_log is not None and not rl_log.empty:
        logs["setup"] = rl_log[rl_log["phase"] == "setup"].reset_index(drop=True)
        logs["gameplay"] = rl_log[rl_log["phase"] == "gameplay"].reset_index(drop=True)
        logs["trade"] = rl_log[rl_log["phase"] == "trade"].reset_index(drop=True)
    else:
        logs = {"setup": pd.DataFrame(), "gameplay": pd.DataFrame(), "trade": pd.DataFrame()}
    return logs

def to_training_dataset_local(rl_log_phase: pd.DataFrame, structure: any, rl_model: RLDecisionModel, normalize_adv: bool = True):
    """
    Build a Keras-ready training dict for PPO from a phase-specific rollout log.

    Parameters
    ----------
    rl_log_phase : pd.DataFrame
        Phase-filtered log with at least the following columns per timestep:
        - "state": np.ndarray of shape (S,) containing the full feature vector.
        - "mask":  np.ndarray of shape (A,) with booleans/0–1 for legal actions.
        - "action": int index in [0, A-1] of the action actually taken.
        - "return": float; discounted sum of future rewards from this timestep.
        - "advantage": float; (pre-normalization) advantage estimate aligned to this timestep.
        - "best_action": (optional, int) argmax of policy at decision time; used only for stats.
        - "policy_probs": (optional, np.ndarray (A,)) behavior policy used to SAMPLE actions
                          (i.e., after any exploration forcing); rows must sum to 1 and have
                          zero mass on illegal actions.
        - "old_action_prob": (optional, float) probability assigned by the behavior policy to
                             the *taken* action (i.e., policy_probs[action]).
    structure : any
        Object exposing `vector_indices` with keys {"nodes","edges","hands"} mapping into
        slices/indices of the flat "state" vector. Also exposes `mask_space_length` (=A).
    rl_model : RLDecisionModel
        Model wrapper exposing `predict_probabilities([x1,x2,x3,mask], ...) -> (probs, values)`.
        Used only when behavior policy is not present in the log; we recompute a consistent
        "old" policy from the current model as a fallback.
    normalize_adv : bool, default True
        If True, advantages are normalized *per round* (z-score within each round id) and then
        clipped for stability before being returned.

    Returns
    -------
    dataset : dict
        A dictionary of numpy arrays that feeds PPO training. Keys:

        - "x_inputs": list of 4 arrays: [x1, x2, x3, masks]
            x1 : np.ndarray, shape (N, |nodes|), float32
            x2 : np.ndarray, shape (N, |edges|), float32
            x3 : np.ndarray, shape (N, |hands|), float32
            masks : np.ndarray, shape (N, A), {0,1}/bool
          These are derived from the "state" and "mask" columns. Rows correspond 1:1 to
          `rl_log_phase` rows. `A = structure.mask_space_length`.

        - "y_policy": np.ndarray, shape (N, A), float32
          One-hot of the *taken* action (from "action"). This is the target used by the PPO
          surrogate (together with advantages and old_action_probs packed into y_true).

        - "y_value": np.ndarray, shape (N, 1), float32
          Training target for the value head, equal to "return" (discounted return).

        - "old_action_probs": np.ndarray, shape (N,), float32
          Probability of the *taken* action under the behavior policy used to generate data.
          If the log provides "policy_probs" and "old_action_prob", those are validated and used
          (preferred). Otherwise, we recompute a consistent "old" policy from `rl_model` on the
          current features and extract probs for the taken action.
          This is the denominator in the PPO ratio `r = π_new(a|s) / π_old(a|s)`.

        - "adv": np.ndarray, shape (N,), float32
          Advantage signal after processing:
            * If `normalize_adv=True`: per-round z-score normalization (mean 0, std 1 within
              each round id) followed by clipping to [-3, 3] (and an extra safety clip to
              [-5, 5] before returning).
            * If `normalize_adv=False`: raw "advantage" is returned unchanged.
          This array should be packed alongside y_policy and old_action_probs to form the
          PPO y_true for the policy loss.

    Behavior policy handling
    ------------------------
    - Preferred path: use logged "policy_probs" (behavior after any exploration forcing) and
      "old_action_prob". We validate that rows sum to 1 and allocate no mass to illegal actions.
    - Fallback path: if behavior is not logged, we compute π_old from `rl_model` on the frozen
      inputs and take π_old(a_taken|s). Note this is *not* the behavior mix if you forced
      exploration during data collection, but it keeps PPO well-defined.

    Invariants & validation
    -----------------------
    - All probability rows (whether from log or model) must sum to 1 within 1e-6 and assign zero
      mass to illegal actions as indicated by "mask".
    - "old_action_probs" values are clipped to [1e-8, 1.0] to avoid numerical issues.
    - Shapes:
        N = len(rl_log_phase)
        A = structure.mask_space_length
      y_policy: (N, A), y_value: (N, 1), old_action_probs: (N,), adv: (N,)

    Notes
    -----
    - The function prints diagnostics (old_action_probs stats, return~round regression,
      advantage normalization stats) to help monitor data quality and round effects.
    - Pack `y_true` for the policy head elsewhere as:
        concat([one_hot_action, adv[:,None], old_action_probs[:,None]], axis=1)
      matching the loss’ expected layout.
    """
    if rl_log_phase is None:
        print("⚠️ Empty log, nothing to convert.")
        return None
    
    dataset = dict([])

    # Convert columns back to arrays to generate x_inputs
    states = np.stack(rl_log_phase["state"].values)
    masks = np.stack(rl_log_phase["mask"].values)
    m = np.asarray(masks, dtype=float)
    masks = np.where(np.isfinite(m) & (m > 0.5), 1.0, 0.0).astype(np.float32)
    x1 = states[:, structure.vector_indices['nodes']]
    x2 = states[:, structure.vector_indices['edges']]
    x3 = states[:, structure.vector_indices['hands']].astype(np.float32)
    dataset["x_inputs"] = [x1, x2, x3, masks]

    # One-hot encode actions. rl_log["action"] is the action actually taken.
    actions = rl_log_phase["action"].values
    num_actions = structure.mask_space_length
    y_policy = np.zeros((len(actions), num_actions), dtype=np.float32)
    y_policy[np.arange(len(actions)), actions] = 1.0
    dataset["y_policy"] = y_policy
    
    # Extract the y-values
    y_value = rl_log_phase["return"].to_numpy().reshape(-1, 1).astype(np.float32)
    dataset["y_value"] = y_value

    # Populate old probs 
    num_actions = structure.mask_space_length
    actions = rl_log_phase["action"].values.astype(int)  # ensure int indexing


    if "policy_probs" in rl_log_phase.columns and "old_action_prob" in rl_log_phase.columns:
        recalculate = False
        old_action_probs = rl_log_phase["old_action_prob"].values.astype(np.float32)
        if old_action_probs.shape[0] != len(actions):
            recalculate = True
        if not np.isfinite(old_action_probs).all():
            recalculate = True
    else: 
        recalculate = True
        
    if not recalculate:
        print("Using logged behavior policy since log contains policy_probs and old_action_prob.")
        # ---- Use logged behavior policy (preferred) ----
        # Stack to (N, A)
        probs_list = rl_log_phase["policy_probs"].to_list()
        probs_mat = np.vstack([np.asarray(p, dtype=np.float32) for p in probs_list])
        if probs_mat.shape[1] != num_actions:
            raise ValueError(f"policy_probs width {probs_mat.shape[1]} != num_actions {num_actions}")

        # After building probs_mat (N, A) and validating it...
        dataset["behavior_probs"] = probs_mat.astype(np.float32)   # for KL_to_behavior

        # Strict sanity checks (no re-masking)
        row_sums = probs_mat.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            bad = np.where(~np.isclose(row_sums, 1.0, atol=1e-6))[0][:10]
            raise ValueError(f"Logged policy_probs rows not normalized; examples rows {bad.tolist()}")

        mask_mat = np.stack(rl_log_phase["mask"].to_numpy()).astype(np.float32)
        if mask_mat.shape != probs_mat.shape:
            raise ValueError(f"mask shape {mask_mat.shape} != policy_probs shape {probs_mat.shape}")

        illegal_mass = (probs_mat * (mask_mat <= 0.5)).sum(axis=1)
        bad_rows = np.where(illegal_mass > 1e-8)[0]
        if bad_rows.size:
            raise ValueError(f"Logged probs allocate mass to illegal actions; example rows {bad_rows[:10].tolist()}")

        old_action_probs = rl_log_phase["old_action_prob"].values.astype(np.float32)
        if old_action_probs.shape[0] != len(actions):
            raise ValueError(f"old_action_prob length {old_action_probs.shape[0]} != number of actions {len(actions)}")
        if not np.isfinite(old_action_probs).all():
            raise ValueError("old_action_probs contains NaN/Inf.")
    else:
        # ---- Fallback: compute pi_old once from current model ----
        print("Recomputing old action probabilities from current model.")
        # IMPORTANT: rl_model.predict_logits_and_value must return *logits* of shape (N, A).
        policy_probs, value_preds = rl_model.predict_probabilities(dataset["x_inputs"], verbose=0)
        # Ensure (N,) for values
        old_value_preds = np.asarray(value_preds).reshape(-1)

        masks_bool   = masks.astype(bool)

        # Basic finiteness/negatives
        if not np.isfinite(policy_probs).all():
            raise ValueError("Probabilities contain NaN/Inf.")
        if (policy_probs < -1e-12).any():
            raise ValueError("Probabilities contain negative values.")

        # Row-wise sums to 1
        row_sums = policy_probs.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            bad = np.where(~np.isclose(row_sums, 1.0, atol=1e-6))[0][:5]
            raise ValueError(f"Prob rows must sum to 1.0; e.g. idx {bad[:1]} sum={row_sums[bad[:1]][0]:.8f}")

        # Illegal mass per row (make sure mask is boolean!)
        illegal_mass_per_row = (policy_probs * (~masks_bool)).sum(axis=1)
        max_illegal = float(illegal_mass_per_row.max())
        if max_illegal > 1e-8:
            bad_i = int(illegal_mass_per_row.argmax())
            raise ValueError(
                f"Illegal actions have non-zero probability mass at row {bad_i}: "
                f"{max_illegal:.3e} (legal mass={policy_probs[bad_i, masks_bool[bad_i]].sum():.3f})"
            )

        # Gather taken-action probs → (N,)
        idx = np.arange(len(actions))
        old_action_probs = np.clip(policy_probs[idx, actions], 1e-8, 1.0)

    # Final shape & finiteness checks
    if old_action_probs.ndim != 1 or old_action_probs.shape[0] != len(actions):
        raise ValueError(f"old_action_probs has wrong shape {old_action_probs.shape}")
    if not np.isfinite(old_action_probs).all():
        raise ValueError("old_action_probs contains NaN/Inf.")
    dataset["old_action_probs"] = old_action_probs

    # === Get insight in the old action probabilities ===
    # print old_action_probs stats
    print(f"Old action probs: mean={old_action_probs.mean():.6f}, std={old_action_probs.std():.6f}, min={old_action_probs.min():.6f}, max={old_action_probs.max():.6f}")
    # print old_action_probs stats when the the taken action was the best action
    best_action_mask = (rl_log_phase["action"].values == rl_log_phase["best_action"].values)
    if best_action_mask.sum() > 0:
        print(f"Old action probs when taken action was best action: mean={old_action_probs[best_action_mask].mean():.6f}, std={old_action_probs[best_action_mask].std():.6f}, min={old_action_probs[best_action_mask].min():.6f}, max={old_action_probs[best_action_mask].max():.6f}")
    else:
        print("No instances where taken action was the best action.")
    # print old_action_probs stats when the taken action was not the best action
    not_best_action_mask = (rl_log_phase["action"].values != rl_log_phase["best_action"].values)
    if not_best_action_mask.sum() > 0:
        print(f"Old action probs when taken action was not best action: mean={old_action_probs[not_best_action_mask].mean():.6f}, std={old_action_probs[not_best_action_mask].std():.6f}, min={old_action_probs[not_best_action_mask].min():.6f}, max={old_action_probs[not_best_action_mask].max():.6f}")
    else:
        print("No instances where taken action was not the best action.")

    # === Get insight in returns vs rounds ===
    # Fit linear regression (degree 1 polynomial)
    rounds_arr = rl_log_phase["round"].values.astype(np.float32)
    returns_arr = rl_log_phase["return"].values.astype(np.float32)
    coeffs = np.polyfit(rounds_arr, returns_arr, deg=1)
    slope, intercept = coeffs[0], coeffs[1]
    # Predictions and R^2
    preds = slope * rounds_arr + intercept
    ss_res = np.sum((returns_arr - preds) ** 2)
    ss_tot = np.sum((returns_arr - returns_arr.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    print(f"Linear regression: return = {slope:.6f} * round + {intercept:.6f} | R^2 = {r2:.4f}")

    # print("Sample old action probs and old value preds:")
    # counter, other_counter = 0,0
    # while counter < 10 and other_counter < len(rl_log_phase):
    #     other_counter += 1
    #     random_number = random.randint(0, len(old_action_probs)-1)
    #     if np.max(old_action_probs[random_number]) < 0.98:
    #         counter += 1
    #         print("Mask:", masks[random_number])
    #         print("Old action probs:", old_action_probs[random_number])
    #         print("Old value preds:", old_value_preds[random_number].flatten())

    # Raw advantages (as stored in your logs)
    adv_raw = rl_log_phase["advantage"].astype(np.float32).values
    print(f"Advantage stats before normalization: "
        f"mean={adv_raw.mean():.4f}, std={adv_raw.std():.4f}, "
        f"min={adv_raw.min():.4f}, max={adv_raw.max():.4f}")

    rl_log_phase["_round_tmp"] = rounds_arr  # attach for groupby

    # --- Per-round z-score normalization ---
    grp = rl_log_phase.groupby("_round_tmp")["advantage"]
    per_round_mean = grp.transform("mean").astype(np.float32)
    per_round_std  = grp.transform("std").astype(np.float32).clip(lower=1e-6)

    advantages = (adv_raw - per_round_mean.values) / per_round_std.values

    # OPTIONAL but common in PPO: clip advantages for stability
    clip_for_ppo = True
    if clip_for_ppo:
        advantages = np.clip(advantages, -3.0, 3.0)

    # Clean up temp column
    rl_log_phase.drop(columns=["_round_tmp"], inplace=True)

    # Sanity prints after per-round normalization
    print(f"Advantage stats AFTER per-round z-score: "
        f"mean={advantages.mean():.4f}, std={advantages.std():.4f}, "
        f"min={advantages.min():.4f}, max={advantages.max():.4f}")

    # Re-run the round regression to confirm the trend is gone
    coeffs = np.polyfit(rounds_arr, advantages, deg=1)
    slope, intercept = coeffs[0], coeffs[1]
    preds = slope * rounds_arr + intercept
    ss_res = np.sum((advantages - preds) ** 2)
    ss_tot = np.sum((advantages - advantages.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    print(f"Linear regression AFTER per-round norm: advantage = {slope:.6f} * round + {intercept:.6f} | R^2 = {r2:.4f}")

    # Hand the normalized (and optionally clipped) advantages to PPO
    # (keep an unclipped copy if you want diagnostics later)
    rl_log_phase["advantage_norm_per_round"] = advantages.astype(np.float32)
    
    # earlier we used positive-only advantages
    # advantages = np.maximum(advantages, 0.0)     # positive-
    # Now we  clip extreme values (5 standard deviations) for stability
    advantages = np.clip(advantages, -5.0, 5.0)
    
    positive_adv_and_greedy_action = (advantages >= 0.0) & (rl_log_phase["action"] == rl_log_phase["best_action"] )
    positive_adv_and_explore_action = (advantages >= 0.0) & (rl_log_phase["action"] != rl_log_phase["best_action"] )
    negative_adv_and_greedy_action = (advantages < 0.0) & (rl_log_phase["action"] == rl_log_phase["best_action"] )
    negative_adv_and_explore_action = (advantages < 0.0) & (rl_log_phase["action"] != rl_log_phase["best_action"] )
    print(f"Positive advantage and greedy action: {positive_adv_and_greedy_action.sum()}")
    print(f"Positive advantage and explore action: {positive_adv_and_explore_action.sum()}")
    print(f"Negative advantage and greedy action: {negative_adv_and_greedy_action.sum()}")
    print(f"Negative advantage and explore action: {negative_adv_and_explore_action.sum()}")

    dataset["adv"] = advantages

    return dataset