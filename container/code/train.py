# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys
import json
import ast
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
from pymysql import converters
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

converters.encoders[np.float64] = converters.escape_float
converters.conversions = converters.encoders.copy()
converters.conversions.update(converters.decoders)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    # MLflow related parameters
    parser.add_argument("--tracking_uri", type=str)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--artifact_location", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--tags", type=str)
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--l1_ratio', type=float, default=0.5)

    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file', type=str, default='wine_quality_train.csv')
    parser.add_argument('--test-file', type=str, default='wine_quality_test.csv')
    parser.add_argument('--features', type=str)  # we ask user to explicitly name features
    parser.add_argument('--target', type=str) # we ask user to explicitly name the target

    args, _ = parser.parse_known_args()

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL if no input train / test file
    logging.info('reading data')
    logging.info('building training and testing datasets')
    logging.info(f"{args.train} {args.train_file} {args.test} {args.test_file}")
    if (args.train_file is None or args.test_file is None or args.features is None or args.target is None):
        logging.info("I'm Here")
        csv_url = (
            'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
        )
        try:
            data = pd.read_csv(csv_url, sep=";")
        except Exception as e:
            logger.exception(
                "Unable to download training & test CSV, check your internet connection. Error: %s", e
            )

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        # The predicted column is "quality" which is a scalar from [3, 9]
        train_x = train.drop(["quality"], axis=1)
        test_x = test.drop(["quality"], axis=1)
        train_y = train[["quality"]]
        test_y = test[["quality"]]
    else:
        # Read data from input CSV file
        train_df = pd.read_csv(os.path.join(args.train, args.train_file))
        test_df = pd.read_csv(os.path.join(args.test, args.test_file))
        logging.info(args.features.split(','))
        train_x = train_df[args.features.split(',')]
        test_x = test_df[args.features.split(',')]
        train_y = train_df[args.target]
        test_y = test_df[args.target]
        

    #alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    #l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    # Train
    
    # Define mlflow experiment
    
    mlflow.set_tracking_uri(args.tracking_uri)
    if mlflow.get_experiment_by_name(args.experiment_name) is None:
        mlflow.create_experiment(args.experiment_name, artifact_location=args.artifact_location)
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run(run_name=args.run_name):
        lr = ElasticNet(alpha=args.alpha, 
                        l1_ratio=args.l1_ratio, 
                        random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
        
        # Logging parameters and metric to mlflow as well as logger
        logging.info(f"Experiment Name: {args.experiment_name}")
        logging.info(f"    Run Name: {args.run_name}")
        logging.info(f"        Elasticnet model: (alpha={args.alpha}, l1_ratio={args.l1_ratio})")
        logging.info(f"            RMSE: {rmse}")
        logging.info(f"            MAE: {mae}")
        logging.info(f"            R2: {r2}")
        logging.info(f"    Tags: {args.tags}") 
        

        mlflow.set_tags(ast.literal_eval(args.tags)) if args.tags else None
        mlflow.log_param("alpha", args.alpha)
        mlflow.log_param("l1_ratio", args.l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)


        # Model registry does not work with file store
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme        
        #if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            
            # Commenting our following code. Register model using UI
        #    mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
        #else:
        #    mlflow.sklearn.log_model(lr, "model")
        
        # Save model
        mlflow.sklearn.log_model(lr, "model")
        logging.info(f"saving model to {args.artifact_location}")
        mlflow.end_run()
