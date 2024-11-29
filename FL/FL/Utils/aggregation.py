import logging
from logging import INFO

import numpy as np
import wandb
from flwr.common.logger import log


class Aggregation:
    def agg_metrics_test(metrics: list, server_round: int, wandb_run) -> dict:
        total_examples = sum([n_examples for n_examples, _ in metrics])

        if "surrogate" in metrics[0][1]:
            loss_evaluation = (
                sum(
                    [n_examples * metric["test_loss"] for n_examples, metric in metrics]
                )
                / total_examples
            )
            fidelity_evaluation = (
                sum(
                    [
                        n_examples * metric["test_fidelity"]
                        for n_examples, metric in metrics
                    ]
                )
                / total_examples
            )

            agg_metrics = {
                "Test Loss": loss_evaluation,
                "Test_Fidelity": fidelity_evaluation,
                "FL Round": server_round,
            }
        elif "explainer" in metrics[0][1]:
            loss_test = (
                sum(
                    [n_examples * metric["test_loss"] for n_examples, metric in metrics]
                )
                / total_examples
            )

            agg_metrics = {
                "Test Loss": loss_test,
                "FL Round": server_round,
            }
        else:
            loss_test = (
                sum([n_examples * metric["loss"] for n_examples, metric in metrics])
                / total_examples
            )
            accuracy_test = (
                sum([n_examples * metric["accuracy"] for n_examples, metric in metrics])
                / total_examples
            )
            f1_test = (
                sum([n_examples * metric["f1_score"] for n_examples, metric in metrics])
                / total_examples
            )

            log(
                INFO,
                f"Test Accuracy: {accuracy_test} - Test Loss {loss_test}",
            )

            agg_metrics = {
                "test_loss": loss_test,
                "Test Accuracy": accuracy_test,
                "FL Round": server_round,
                "Test F1": f1_test,
            }
        if wandb_run:
            wandb_run.log(agg_metrics)

        return agg_metrics

    def agg_metrics_evaluation(metrics: list, server_round: int, wandb_run) -> dict:
        total_examples = sum([n_examples for n_examples, _ in metrics])

        if "surrogate" in metrics[0][1]:
            loss_evaluation = (
                sum(
                    [
                        n_examples * metric["validation_loss"]
                        for n_examples, metric in metrics
                    ]
                )
                / total_examples
            )
            fidelity_evaluation = (
                sum(
                    [
                        n_examples * metric["validation_fidelity"]
                        for n_examples, metric in metrics
                    ]
                )
                / total_examples
            )

            agg_metrics = {
                "validation_loss": loss_evaluation,
                "Validation_Fidelity": fidelity_evaluation,
                "FL Round": server_round,
            }
        elif "explainer" in metrics[0][1]:
            loss_evaluation = (
                sum(
                    [
                        n_examples * metric["validation_loss"]
                        for n_examples, metric in metrics
                    ]
                )
                / total_examples
            )

            agg_metrics = {
                "validation_loss": loss_evaluation,
                "FL Round": server_round,
            }
        else:
            loss_evaluation = (
                sum([n_examples * metric["loss"] for n_examples, metric in metrics])
                / total_examples
            )
            accuracy_evaluation = (
                sum([n_examples * metric["accuracy"] for n_examples, metric in metrics])
                / total_examples
            )
            f1_validation = (
                sum([n_examples * metric["f1_score"] for n_examples, metric in metrics])
                / total_examples
            )

            agg_metrics = {
                "validation_loss": loss_evaluation,
                "Validation_Accuracy": accuracy_evaluation,
                "FL Round": server_round,
                "Validation F1": f1_validation,
            }
        if wandb_run:
            wandb_run.log(agg_metrics)

        return agg_metrics

    def agg_metrics_train(metrics: list, server_round: int, fed_dir, wandb_run) -> dict:
        # Collect the losses logged during each epoch in each client
        total_examples = sum([n_examples for n_examples, _ in metrics])
        losses = []
        losses_with_regularization = []
        accuracies = []
        surrogate = False
        explainer = False
        fidelity_list = []
        for n_examples, node_metrics in metrics:
            if "surrogate" in node_metrics:
                surrogate = True
                losses.append(n_examples * node_metrics["train_loss"])
                fidelity_list.append(n_examples * node_metrics["train_fidelity"])
                client_id = node_metrics["cid"]
            elif "explainer" in node_metrics:
                explainer = True
                losses.append(n_examples * node_metrics["train_loss"])
                client_id = node_metrics["cid"]
            else:
                losses.append(n_examples * node_metrics["train_loss"])
                accuracies.append(n_examples * node_metrics["train_accuracy"])
                client_id = node_metrics["cid"]

                # Create the dictionary we want to log. For some metrics we want to log
                # we have to check if they are present or not.
                to_be_logged = {
                    "FL Round": server_round,
                }

                if wandb_run:
                    wandb_run.log(
                        to_be_logged,
                    )

        log(
            INFO,
            f"Train Accuracy: {sum(accuracies) / total_examples} - Train Loss {sum(losses) / total_examples}",
        )
        if surrogate:
            agg_metrics = {
                "Train Loss": sum(losses) / total_examples,
                "Train Fidelity": sum(fidelity_list) / total_examples,
                "FL Round": server_round,
            }
        elif explainer:
            agg_metrics = {
                "Train Loss": sum(losses) / total_examples,
                "FL Round": server_round,
            }
        else:
            agg_metrics = {
                "Train Loss": sum(losses) / total_examples,
                "Train Accuracy": sum(accuracies) / total_examples,
                "Train Loss with Regularization": sum(losses_with_regularization)
                / total_examples,
                "FL Round": server_round,
            }
        if wandb_run:
            wandb_run.log(agg_metrics)

        return agg_metrics
