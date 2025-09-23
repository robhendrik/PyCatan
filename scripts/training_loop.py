import run_evaluation_tournament as eval
import RL_training_loop_script_for_ec2_new_parallel as rl
import argparse
import os



for round in range(24):
    print(f"=== Round {round+1}/24 ===")
    rl.main(argparse.Namespace(working_model="working_model.keras", train_games=96, gamma=0.99, n_jobs=os.cpu_count()))  # train for 96 games each round
    eval.main(argparse.Namespace(model_name="working_model.keras", identifier=round, verbose=False))