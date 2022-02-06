#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights and Biases
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    logger.info("### Starting the data cleaning process ###")
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    logger.info(f"Loading file from the path: {artifact_local_path}")
    df = pd.read_csv(artifact_local_path)

    # Drop outliers
    logger.info("Dropping outliers in price from the dataset")
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.info("Converting the last_review column to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    #Drop outliers
    logger.info("Dropping outliers in longitude from the dataset")
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    logger.info("Saving the dataset")
    df.to_csv("clean_sample.csv", index=False)

    logger.info("Loading the dataset to wandb")
    artifact = wandb.Artifact(
     args.output_artifact,
     type=args.output_type,
     description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")


    parser.add_argument(
        "--input_artifact",
        type=str,
        help="input artifact name loaded in wandb",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="output artifact name loaded in wandb",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="type of output to be displayed in wandb",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="output description to be displayed in wandb",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="minimum price to consider in the data cleaning",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="max price to consider in the data cleaning",
        required=True
    )

    args = parser.parse_args()

    go(args)
