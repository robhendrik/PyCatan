import run_evaluation_tournament as eval
import RL_training_loop_script_for_ec2_new_parallel as rl
import argparse
import os
import boto3

s3 = boto3.client("s3")
bucket_name = "pycatanbucket"


rounds = 2
for round in range(rounds):
    print(f"=== Round {round+1}/rounds ===")
    rl.main(argparse.Namespace(working_model="working_model.keras", train_games=24, gamma=0.99, n_jobs=os.cpu_count()))  # train for 24 games each round
    eval.main(argparse.Namespace(model_name="working_model.keras", identifier=round, verbose=True))
    s3.upload_file("working_model.keras", bucket_name, f"models/working_model_{round}.keras")