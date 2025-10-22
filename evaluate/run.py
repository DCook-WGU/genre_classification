#!/usr/bin/env python
import argparse
import itertools
import logging
import pandas as pd
import wandb
import mlflow.sklearn
import matplotlib.pyplot as plt
#from sklearn.metrics import roc_auc_score, plot_confusion_matrix
from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="test")

    logger.info("Downloading and reading test artifact")
    test_data_path = run.use_artifact(args.test_data).file()
    df = pd.read_csv(test_data_path, low_memory=False)

    # Extract the target from the features
    logger.info("Extracting target from dataframe")
    X_test = df.copy()
    y_test = X_test.pop("genre")

    logger.info("Downloading and reading the exported model")
    model_export_path = run.use_artifact(args.model_export).download()

    pipe = mlflow.sklearn.load_model(model_export_path)

    used_columns = list(itertools.chain.from_iterable([x[2] for x in pipe['preprocessor'].transformers]))
    pred_proba = pipe.predict_proba(X_test[used_columns])

    logger.info("Scoring")
    score = roc_auc_score(y_test, pred_proba, average="macro", multi_class="ovo")

    run.summary["AUC"] = score

    logger.info("Computing confusion matrix")
    
    '''
    fig_cm, sub_cm = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(
        pipe,
        X_test[used_columns],
        y_test,
        ax=sub_cm,
        normalize="true",
        values_format=".1f",
        xticks_rotation=90,
    )
    
    fig_cm.tight_layout()

    run.log(
        {
            "confusion_matrix": wandb.Image(fig_cm)
        }
    )
    '''

    ### Modernizing confusion matrix

    fig_cm, sub_cm = plt.subplots(figsize=(10, 10))

    # New API (replaces plot_confusion_matrix)
    disp = ConfusionMatrixDisplay.from_estimator(
        pipe,
        X_test[used_columns],
        y_test,
        normalize="true",   # same behavior as before
        ax=sub_cm,
    )

    # Match the old formatting
    sub_cm.set_title("Confusion Matrix")
    sub_cm.set_xticklabels(sub_cm.get_xticklabels(), rotation=90)

    # Re-format cell text to one decimal place (like values_format=".1f")
    for text in disp.text_.ravel():
        try:
            text.set_text(f"{float(text.get_text()):.1f}")
        except ValueError:
            pass  # skip labels that aren't numbers
        text.set_fontsize(8)

    fig_cm.tight_layout()

    # Log to W&B
    run.log({"confusion_matrix": wandb.Image(fig_cm)})

    plt.close(fig_cm)  # tidy up figure resources

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the provided model on the test artifact",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--model_export",
        type=str,
        help="Fully-qualified artifact name for the exported model to evaluate",
        required=True,
    )

    parser.add_argument(
        "--test_data",
        type=str,
        help="Fully-qualified artifact name for the test data",
        required=True,
    )

    args = parser.parse_args()

    go(args)
