from pathlib import Path
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, get_origin, get_args, Tuple
import argparse

def get_repo_root(target_name: str = "PyCatan") -> Path:
    """
    Return the project root by walking up from this file until:
      - a directory is named `target_name`, OR
      - a common project marker is found (.git or pyproject.toml)
    """
    cur = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
    for parent in [cur, *cur.parents]:
        # 1) Directory-name match (case-insensitive to be Windows-friendly)
        if parent.name.lower() == target_name.lower():
            return parent
        # 2) Common project markers as fallback (works even if folder is renamed)
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(
        f"Project root not found (no folder named '{target_name}' and no .git/pyproject.toml up the tree)."
    )

def print_args(args):
    """
    Print all training and configuration parameters from an Args instance.
    """
    print("\n=== TRAINING CONFIGURATION ===")
    print(f"Phase to train:              {args.phase_to_train}")
    print(f"Training rounds:             {args.training_rounds}")
    print(f"Current round:               {args.training_round}")
    print(f"Tournaments per round:       {args.tournaments_per_round}")
    print(f"Games per tournament:        {args.games_per_tournament}")

    print("\n=== EVALUATION SETTINGS ===")
    print(f"Evaluation games:            {args.number_of_games_in_evaluation_tournament}")
    print(f"Evaluation interval:         {args.evaluation_tournament_interval}")
    print(f"Tournament log filename:     {args.filename_for_tournament_log}")

    print("\n=== TRAINING PARAMETERS ===")
    print(f"Entropy coefficient:         {args.entropy_coef}")
    print(f"Value coefficient:           {args.value_coef}")
    print(f"Learning rate:               {args.learning_rate}")
    print(f"Clip norm:                   {args.clipnorm}")
    print(f"Epsilon:                     {args.epsilon}")
    print(f"Clip ratio:                  {args.clip_ratio}")
    print(f"Epochs:                      {args.epochs}")
    print(f"Batch size:                  {args.batch_size}")
    print(f"Target KL:                   {args.target_kl}")
    print(f"LR backoff:                  {args.lr_backoff}")
    print(f"Discount factor (gamma):     {args.gamma}")
    print(f"Behavioral cloning beta:     {args.bc_beta_HARDCODED}")

    print("\n=== MODEL SETTINGS ===")
    print(f"S3 bucket name:              {args.s3_bucket_name}")
    print(f"Starting model (TRADE):      {args.starting_model_name_trade}")
    print(f"Starting model (GAMEPLAY):   {args.starting_model_name_gameplay}")
    print(f"Starting model (SETUP):      {args.starting_model_name_setup}")
    print(f"Working model local:         {args.model_name_on_local}")
    print(f"Working model on S3:         {args.filename_for_storing_model_on_s3}")

    print("\n=== DATASET SETTINGS ===")
    print(f"Files to download:           {args.number_of_files_to_download}")
    print(f"Stable datasets to mix:      {args.number_of_stable_datasets_to_mix_in}")
    print(f"Previous datasets to mix:    {args.number_of_previous_datasets_to_mix_in}")
    print(f"RL log filename pattern:     {args.filename_for_storing_rl_log_on_s3}")
    print(f"Full list of S3 filenames:   {args.full_list_of_filenames_on_s3}")

    print("\n=== PARALLELIZATION ===")
    print(f"Number of workers:           {args.number_of_workers}")
    print(f"Games per worker:            {args.games_per_worker}")

    print("\n=== OTHER ===")
    print(f"Fixed player order:          {args.fixed_player_order}")
    print("============================================\n")

from dataclasses import dataclass

@dataclass
class Args:
    phase_to_train: str = 'TRADE' # 'TRADE' or 'GAMEPLAY' or 'SETUP'
    training_rounds: int = 4 # for training
    training_round: int = 0 # first round, if this is 10 and training rounds is 20 we run rounds from 10 to 29
    tournaments_per_round: int = 2 # 4 # for generating training data
    games_per_tournament: int = 12 #24 # for generating training data

    # Evaluation tournament settings
    number_of_games_in_evaluation_tournament: int = 3 # 24 # for evaluating performance per training round
    evaluation_tournament_interval: int = 1 # run evaluation tournament every N rounds
    filename_for_tournament_log: str = "TEMP_tournament_2_log_identifier.csv" # identifier will be replaced with training round number # data from evaluation tournaments

    # training parameters
    entropy_coef: float = 0.12
    value_coef: float = 0.5
    learning_rate: float = 1e-4 # 5e-6
    clipnorm: float = 0.5
    epsilon: float = 1e-7
    clip_ratio: float = 0.2
    epochs: int = 20
    batch_size: int = 128
    loss_weights_output: float = 1.0
    loss_weights_value: float = 0.1
    target_kl: float = 0.25
    lr_backoff: float = 0.8
    bc_beta_HARDCODED: float = 0.00  # try 0.05–0.1

    # Discount factor
    gamma: float = 0.99 # discount factor for future rewardstau = args.tau # temperature for sharpening policy probabilities
    tau: float = 0.1 # temperature for sharpening policy probabilities

    # initial models
    s3_bucket_name: str = "pycatanbucket"
    starting_model_name_gameplay: str = "bootstrap/rl_decision_model_trained_gameplay.keras"
    starting_model_name_setup: str = "bootstrap/rl_decision_model_mimic_setup.keras"
    starting_model_name_trade: str = "models/working_model_4_1.keras" # bootstrap/rl_decision_model_trained_trade.keras
    
    # working model (for phase to be trained)
        # model will be stored on S3 as models/working_model_3_identifier.keras, identifier will be replaced with training round number
    filename_for_storing_model_on_s3: str = "models/working_model_4_identifier.keras"
    model_name_on_local: str = "working_model.keras"
    
    # bootstrapping setings
    tournament_indicators: tuple[int, ...] = (0, 1) # used to identify tournaments when generating data
    meta_indicators: tuple[int, ...] = (0,) # used to identify meta indicators when generating data
    filename_for_bootstrapping_logs: str = f"tutorial_metameta_indicator_tournamenttournament_indicator_rl_log.pkl"
    directory_on_s3_for_logs: str = f"bootstrap/"
    local_directory_for_logs: str =f"bootstrap/"

    # initial data for stabilizing training, download pre-generated data from S3
    number_of_files_to_download: int = 2 # 12
    number_of_stable_datasets_to_mix_in: int = 1
    number_of_previous_datasets_to_mix_in: int = 1
        # logs will be stored on S3 as data/rl_log_4_ppo_round_0.pkl, data/rl_log_4_ppo_round_1.pkl, ...
    filename_for_storing_rl_log_on_s3: str = "rl_log_4_ppo_round_identifier.pkl" # identifier will be replaced with training round number
    full_list_of_filenames_on_s3: tuple[str] = ("data/rl_log_4_ppo_round_0.pkl","data/rl_log_4_ppo_round_1.pkl")

    # parallelization settings, only relevant if exploiting multi-core setup
    number_of_workers: int = 8
    games_per_worker: int = 24

    # Varying or fixed permutation pf player order
    fixed_player_order: bool = False # Random is more representant, fixed will give less variance and show progress more quickly

def _str2bool(v: str) -> bool:
    if isinstance(v, bool): return v
    v = v.lower()
    if v in {"1","true","t","yes","y","on"}: return True
    if v in {"0","false","f","no","n","off"}: return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got '{v}'")

def _is_seq_type(t): 
    return get_origin(t) in (list, List, tuple, Tuple)

def parse_args(argv=None) -> Args:
    parser = argparse.ArgumentParser(add_help=True, allow_abbrev=False)

    for f in fields(Args):
        name, default, typ = f.name, f.default, f.type
        flag = f"--{name.replace('_','-')}"   # CLI-friendly flag
        kwargs = dict(dest=name)              # write into dataclass field name

        if typ is bool:
            parser.add_argument(flag, type=_str2bool, nargs="?", const=True, default=default, **kwargs)
            continue

        if typ is Path:
            parser.add_argument(flag, type=Path, default=default, **kwargs)
            continue

        if _is_seq_type(typ):
            elem_type = get_args(typ)[0] if get_args(typ) else str
            # parse N values: --tournament-indicators 0 1 5
            parser.add_argument(flag, nargs="+", type=elem_type,
                                default=list(default) if default is not None else None,
                                **kwargs)
            continue

        # scalar (str, int, float, etc.)
        parser.add_argument(flag, type=typ, default=default, **kwargs)

    ns = parser.parse_args(argv)
    kw = vars(ns)

    # Cast lists back to tuples for tuple-typed fields
    for f in fields(Args):
        if get_origin(f.type) in (tuple, Tuple) and isinstance(kw.get(f.name), list):
            kw[f.name] = tuple(kw[f.name])

    # Provide default if your S3 list is still None (adjust if you keep it as tuple)
    if kw.get("full_list_of_filenames_on_s3") is None:
        kw["full_list_of_filenames_on_s3"] = (
            "data/rl_log_4_ppo_round_0.pkl",
            "data/rl_log_4_ppo_round_1.pkl",
        )

    return Args(**kw)