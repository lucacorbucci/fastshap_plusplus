import copy
import os

# import from parent level
import sys

import dill
import numpy as np
import shap  # https://github.com/slundberg/shap
import shapreg  # https://github.com/iancovert/shapley-regression
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append("../")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import shap  # https://github.com/slundberg/shap
import shapreg  # https://github.com/iancovert/shapley-regression
import torch
import torch.nn as nn
import torch.nn.functional as F
from aix360.metrics.local_metrics import faithfulness_metric
from scipy.stats import kendalltau, sem, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fastshap import FastSHAP, KLDivLoss, Surrogate
from fastshap.utils import MaskLayer1d

sys.path.append("../")
import os
import sys

import torch
import torch.nn as nn
from Client.client_explainer import prepare_dataset_for_explainer_FL
from metrics import agreement_fraction, pairwise_rank_agreement, rankcorr
from Utils.utils import Utils

from fastshap.surrogate_dp import SurrogateDP

sys.path.append("../")
import warnings
from pathlib import Path

from fastshap.fastshap_dp import (
    FastSHAP,
    calculate_grand_coalition_FL,
    validate_FL,
)

# ignore warnings
warnings.filterwarnings("ignore")


def prepare_data_FL(
    test_clients,
    batch_size,
    dataset_name,
    num_workers,
    fed_dir,
    device,
    num_features,
    surrogate,
    seed=42,
):
    samples, targets = [], []
    for client in test_clients:
        data = Utils.get_dataset(
            path_to_data=Path(fed_dir),
            cid=client,
            dataset=dataset_name,
            partition="train",
        )
        for item in data:
            X, _, y = item
            samples.append(X)
            targets.append(y)

    return samples, targets


def load_data(
    private_model: bool,
    private_surrogate: bool,
    explainer_privacy_levels: list,
    base_path: str,
    model_name: str,
    surrogate_name: str,
    explainer_name: str,
    device: str,
    num_features: int,
    eps_bb: str = "",
    no_dp_explainer=None,
):
    loaded_data = {}
    loaded_data["private_model"] = private_model
    loaded_data["private_surrogate"] = private_surrogate
    private_model_str = "private_model_" if private_model else ""

    if os.path.isfile(
        f"{base_path}bb/{model_name}"
        + ("_NO_DP" if not private_model else f"_{eps_bb}")
        + ".pth"
    ):
        print("Loading saved model")
        model = torch.load(
            f"{base_path}bb/{model_name}"
            + ("_NO_DP" if not private_model else f"_{eps_bb}")
            + ".pth"
        ).to("cpu")
        model_aix = aix_model(model)
    else:
        print(
            f"Model not found: {base_path}bb/{model_name}"
            + ("_NO_DP" if not private_model else f"_{eps_bb}")
            + ".pth"
        )

    if os.path.isfile(f"{base_path}/surrogate/{surrogate_name}" + ".pth"):
        print(f"Loading saved surrogate model {surrogate_name}.pth")
        surr = torch.load(f"{base_path}/surrogate/{surrogate_name}" + ".pth").to(device)
        surrogate = Surrogate(surr, num_features)
    else:
        print(
            f"Surrogate model not found: {base_path}/surrogate/{surrogate_name}"
            + ".pth"
        )
    # if no_dp_explainer:
    #     print("OK")
    #     if os.path.isfile(
    #         f"{base_path}/surrogate/{surrogate_name}_NO_DP.pth"
    #     ):
    #         print("Loading saved surrogate model without privacy")
    #         surr = torch.load(
    #             f"{base_path}/surrogate/{surrogate_name}_NO_DP.pth"
    #         ).to(device)
    #         surrogate = Surrogate(surr, num_features)
    #     else:
    #         raise ValueError("No DP explainer not found")

    loaded_data["model"] = model
    loaded_data["model_aix"] = model_aix
    loaded_data["surrogate"] = surrogate

    if no_dp_explainer:
        if os.path.isfile(f"{base_path}/explainer/{no_dp_explainer}.pth"):
            print(f"Loading saved explainer model without DP {no_dp_explainer}")
            explainer = torch.load(f"{base_path}/explainer/{no_dp_explainer}.pth").to(
                device
            )
            fastshap = FastSHAP(
                explainer,
                surrogate,
                normalization="none",
                link=nn.Softmax(dim=-1),
                num_features=num_features,
            )
            loaded_data[f"explainer_NO_DP"] = fastshap
        else:
            print(
                f"Explainer model not found: {base_path}/explainer/{no_dp_explainer}.pth"
            )
    else:
        raise ValueError("No DP explainer not found")

    for eps in explainer_privacy_levels:
        # DP 0.5
        if os.path.isfile(f"{base_path}/explainer/{explainer_name}_{eps}.pth"):
            print("Loading saved explainer model")
            explainer = torch.load(
                f"{base_path}/explainer/{explainer_name}_{eps}.pth"
            ).to(device)
            fastshap = FastSHAP(
                explainer,
                surrogate,
                normalization="none",
                link=nn.Softmax(dim=-1),
                num_features=num_features,
            )
            loaded_data[f"explainer_{eps}"] = fastshap
        else:
            print(
                f"Explainer model not found: {base_path}/explainer/{explainer_name}_{private_model_str}{eps}.pth"
            )
    return loaded_data


# Faithfulness evaluation
class aix_model:
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        x = torch.Tensor(x)
        return self.model(x).argmax(dim=1)

    def predict_proba(self, x):
        # since the activation function of the last layer is LogSoftmax
        # we need to apply the exponential to the output of the model
        # cast x to be a Tensor
        x = torch.Tensor(x)
        return torch.nn.functional.softmax(self.model(x)).detach().numpy()


def compute_faithfulness(x, y, fastshap_explanation, model, base_value):
    x = x[0]
    fastshap_explanation = np.array(torch.tensor(fastshap_explanation).cpu())

    faithfulness = faithfulness_metric(
        model=model,
        x=np.array(x),
        coefs=fastshap_explanation,
        base=base_value * np.ones(shape=fastshap_explanation.shape[0]),
    )
    return faithfulness


def process_explainer(
    explanation_file_name, explainer, model, x, y, index, base_value, base_path
):
    if not os.path.isfile(
        f"{base_path}/explanations/{explanation_file_name}_{index}.pt"
    ):
        fastshap_explanation = explainer.shap_values(x)[0][:, y]
        torch.save(
            fastshap_explanation,
            f"{base_path}/explanations/{explanation_file_name}_{index}.pt",
        )
    else:
        fastshap_explanation = torch.load(
            f"{base_path}/explanations/{explanation_file_name}_{index}.pt"
        )

    if not os.path.isfile(
        f"{base_path}/faithfulness/{explanation_file_name}_{index}_{base_value}.pt"
    ):
        faithfulness = compute_faithfulness(
            x,
            y,
            fastshap_explanation,
            model,
            base_value=base_value,
        )
        torch.save(
            faithfulness,
            f"{base_path}/faithfulness/{explanation_file_name}_{index}_{base_value}.pt",
        )
    else:
        faithfulness = torch.load(
            f"{base_path}/faithfulness/{explanation_file_name}_{index}_{base_value}.pt"
        )
    return fastshap_explanation, faithfulness


class Args:
    def __init__(self):
        self.dataset_name = "adult"
        self.sweep = True


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_explanations(
    model_predictions: list,
    # model_predictions_softmax: list,
    X_test,
    loaded_data: dict,
    base_value: float,
    base_path: str,
    store_path: str,
    explainer_privacy_levels: list,
):
    for privacy_level in explainer_privacy_levels:
        print("Privacy level: ", privacy_level)
        faithfulness = []
        explanations = []
        for index, (x, y) in enumerate(zip(X_test, model_predictions)):
            y = 0 if not y else 1
            x = np.array([list(x)])
            shap_NO_DP, faithfulness_NO_DP = process_explainer(
                f"private_model_{loaded_data['private_model']}_surrogate_{loaded_data['private_surrogate']}_explainer_{privacy_level}",
                loaded_data[f"explainer_{privacy_level}"],
                loaded_data["model_aix"],
                x,
                y,
                index,
                base_value,
                base_path=store_path,
            )
            explanations.append(copy.deepcopy(shap_NO_DP))
            faithfulness.append(faithfulness_NO_DP)
            loaded_data[f"explanations_{privacy_level}"] = explanations
            loaded_data[f"faithfulness_{privacy_level}"] = faithfulness


def get_model_predictions(model, X_test):
    model_predictions = []
    for x in X_test:
        model_prediction = model(torch.Tensor(x).unsqueeze(0)).argmax()
        model_predictions.append(model_prediction)
    return model_predictions


def compute_metrics_single_explainer(
    X_test,
    surrogate_NO_DP,
    surrogate_DP,
    selected_features,
    all_features,
    top_k=None,
):
    metrics = {}
    L2_distances = []
    spearman_correlation = []
    cosine_similarity = []
    kendall_tau = []
    for index in range(len(X_test)):
        selected_indexes = [
            all_features.index(feature) for feature in selected_features
        ]

        explanation_NO_DP = surrogate_NO_DP["explanations_NO_DP"][index][
            selected_indexes
        ]

        explanation_DP = surrogate_DP[f"explanations_NO_DP"][index][selected_indexes]

        L2_distances.append(np.linalg.norm(explanation_NO_DP - explanation_DP))
        coef, p = spearmanr(explanation_DP, explanation_NO_DP)
        spearman_correlation.append(coef)
        cosine_similarity.append(
            np.dot(explanation_DP, explanation_NO_DP)
            / (np.linalg.norm(explanation_DP) * np.linalg.norm(explanation_NO_DP))
        )
        tau, p_value = kendalltau(explanation_DP, explanation_NO_DP)
        kendall_tau.append(tau)

    feature_agreement = agreement_fraction(
        attrA=np.array(surrogate_NO_DP["explanations_NO_DP"]),
        attrB=np.array(surrogate_DP[f"explanations_NO_DP"]),
        k=5,
        metric="feature",
    )
    rank_agreement = agreement_fraction(
        attrA=np.array(surrogate_NO_DP["explanations_NO_DP"]),
        attrB=np.array(surrogate_DP[f"explanations_NO_DP"]),
        k=5,
        metric="rank",
    )
    sign_agreement = agreement_fraction(
        attrA=np.array(surrogate_NO_DP["explanations_NO_DP"]),
        attrB=np.array(surrogate_DP[f"explanations_NO_DP"]),
        k=5,
        metric="sign",
    )
    signed_rank_agreement = agreement_fraction(
        attrA=np.array(surrogate_NO_DP["explanations_NO_DP"]),
        attrB=np.array(surrogate_DP[f"explanations_NO_DP"]),
        k=5,
        metric="signedrank",
    )
    pairwise_rank_agreement_score = pairwise_rank_agreement(
        attrA=np.array(surrogate_NO_DP["explanations_NO_DP"]),
        attrB=np.array(surrogate_DP[f"explanations_NO_DP"]),
    )
    rank_corr = rankcorr(
        attrA=np.array(surrogate_NO_DP["explanations_NO_DP"]),
        attrB=np.array(surrogate_DP[f"explanations_NO_DP"]),
    )

    metrics[f"L2"] = np.mean(L2_distances)
    metrics[f"spearman"] = np.mean(spearman_correlation)
    metrics[f"cosine"] = np.mean(cosine_similarity)
    metrics[f"kendall"] = np.mean(kendall_tau)
    metrics[f"feature_agreement"] = np.mean(feature_agreement)
    metrics[f"rank_agreement"] = np.mean(rank_agreement)
    metrics[f"sign_agreement"] = np.mean(sign_agreement)
    metrics[f"signed_rank_agreement"] = np.mean(signed_rank_agreement)
    metrics[f"pairwise_rank_agreement"] = np.mean(pairwise_rank_agreement_score)
    metrics[f"rank_corr"] = np.mean(rank_corr)

    metrics[f"L2_std"] = np.std(L2_distances)
    metrics[f"spearman_std"] = np.std(spearman_correlation)
    metrics[f"cosine_std"] = np.std(cosine_similarity)
    metrics[f"kendall_std"] = np.std(kendall_tau)
    metrics[f"feature_agreement_std"] = np.std(feature_agreement)
    metrics[f"rank_agreement_std"] = np.std(rank_agreement)
    metrics[f"sign_agreement_std"] = np.std(sign_agreement)
    metrics[f"signed_rank_agreement_std"] = np.std(signed_rank_agreement)
    metrics[f"pairwise_rank_agreement_std"] = np.std(pairwise_rank_agreement_score)
    metrics[f"rank_corr_std"] = np.std(rank_corr)

    return metrics


def plot_base_explainer_shap_values(
    loaded_data, explainer_privacy_levels, X_test, feature_names, title, explainer_name
):
    explanations = []
    for index in range(len(X_test)):
        fastshap_NO_DP = loaded_data[explainer_name][index]
        # compute the absolute values of the shap values
        fastshap_NO_DP = np.abs(fastshap_NO_DP)
        explanations.append(fastshap_NO_DP)

    # mean of the absolute values of the shap values
    mean_explanations = np.mean(explanations, axis=0)
    # sort the mean_explanations and the corresponding feature names
    sorted_indices = np.argsort(mean_explanations)[::-1]
    print(len(sorted_indices))
    print(len(feature_names))
    # plot the mean shap values with the corresponding feature names in sorted indices
    plt.figure(figsize=(10, 6))
    plt.bar(
        [feature_names[i] for i in sorted_indices], mean_explanations[sorted_indices]
    )
    plt.xticks(rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Mean SHAP Value")
    plt.title(title)
    # plt.show()
    plt.savefig(f"{title}.png")


def compute_metrics(
    X_test,
    loaded_data,
    explainer_privacy_levels,
    selected_features,
    all_features,
    top_k=None,
):
    metrics = {}
    for privacy_level in explainer_privacy_levels:
        L2_distances = []
        spearman_correlation = []
        cosine_similarity = []
        kendall_tau = []
        for index in range(len(X_test)):
            selected_indexes = [
                all_features.index(feature) for feature in selected_features
            ]

            explanation_NO_DP = loaded_data["explanations_NO_DP"][index][
                selected_indexes
            ]

            explanation_DP = loaded_data[f"explanations_{privacy_level}"][index][
                selected_indexes
            ]

            L2_distances.append(np.linalg.norm(explanation_NO_DP - explanation_DP))
            coef, p = spearmanr(explanation_DP, explanation_NO_DP)
            spearman_correlation.append(coef)
            cosine_similarity.append(
                np.dot(explanation_DP, explanation_NO_DP)
                / (np.linalg.norm(explanation_DP) * np.linalg.norm(explanation_NO_DP))
            )
            tau, p_value = kendalltau(explanation_DP, explanation_NO_DP)
            kendall_tau.append(tau)

        feature_agreement = agreement_fraction(
            attrA=np.array(loaded_data["explanations_NO_DP"]),
            attrB=np.array(loaded_data[f"explanations_{privacy_level}"]),
            k=5,
            metric="feature",
        )
        rank_agreement = agreement_fraction(
            attrA=np.array(loaded_data["explanations_NO_DP"]),
            attrB=np.array(loaded_data[f"explanations_{privacy_level}"]),
            k=5,
            metric="rank",
        )
        sign_agreement = agreement_fraction(
            attrA=np.array(loaded_data["explanations_NO_DP"]),
            attrB=np.array(loaded_data[f"explanations_{privacy_level}"]),
            k=5,
            metric="sign",
        )
        signed_rank_agreement = agreement_fraction(
            attrA=np.array(loaded_data["explanations_NO_DP"]),
            attrB=np.array(loaded_data[f"explanations_{privacy_level}"]),
            k=5,
            metric="signedrank",
        )
        pairwise_rank_agreement_score = pairwise_rank_agreement(
            attrA=np.array(loaded_data["explanations_NO_DP"]),
            attrB=np.array(loaded_data[f"explanations_{privacy_level}"]),
        )
        rank_corr = rankcorr(
            attrA=np.array(loaded_data["explanations_NO_DP"]),
            attrB=np.array(loaded_data[f"explanations_{privacy_level}"]),
        )

        metrics[f"L2_{privacy_level}"] = L2_distances
        metrics[f"spearman_{privacy_level}"] = spearman_correlation
        metrics[f"cosine_{privacy_level}"] = cosine_similarity
        metrics[f"kendall_{privacy_level}"] = kendall_tau
        metrics[f"feature_agreement_{privacy_level}"] = feature_agreement
        metrics[f"rank_agreement_{privacy_level}"] = rank_agreement
        metrics[f"sign_agreement_{privacy_level}"] = sign_agreement
        metrics[f"signed_rank_agreement_{privacy_level}"] = signed_rank_agreement
        metrics[f"pairwise_rank_agreement_{privacy_level}"] = (
            pairwise_rank_agreement_score
        )
        metrics[f"rank_corr_{privacy_level}"] = rank_corr
    return metrics


test_nodes = ["25", "23", "19", "11", "4", "45", "26", "9", "29", "16"]
explainer_privacy_levels = ["NO_DP", "DP_01", "DP_05", "DP_1", "DP_2", "DP_5"]
bb_privacy_levels = ["NO_DP"]


base_path = "../../../artifacts/employment_NO_DP_surrogate/"
store_path = "/raid/lcorbucci/folktables/employment_data_reduced/"
surrogate_name = "surrogate_DP_1"
surrogate_NO_DP_name = "employment_surrogate_NO_DP"
fed_dir = "/raid/lcorbucci/folktables/employment_data_reduced/federated"
dataset_name = "employment"
num_features = 17
top_k_private_bb = [
    "feature_2",
    "feature_9",
    "feature_6",
    "feature_1",
]
metrics_folder = "../../../artifacts/employment_NO_DP_surrogate/metrics/"
feature_names = [f"feature_{i}" for i in range(1, num_features+1)]

loaded_data_surrogate_DP = load_data(
    private_model=True,
    private_surrogate=True,
    explainer_privacy_levels=explainer_privacy_levels[1:],
    base_path=base_path,
    model_name="bb_DP_1",
    surrogate_name=surrogate_name,
    explainer_name="explainer",
    device="cuda",
    num_features=num_features,
    no_dp_explainer="explainer_NO_DP",
)

loaded_data_surrogate_NO_DP = load_data(
    private_model=True,
    private_surrogate=False,
    explainer_privacy_levels=explainer_privacy_levels[1:],
    base_path=base_path,
    model_name="bb_DP_1",
    surrogate_name=surrogate_NO_DP_name,
    explainer_name="explainer",
    device="cuda",
    num_features=num_features,
    no_dp_explainer="explainer_NO_DP_surrogate_NO_DP",
)

X_test, y_test = prepare_data_FL(
    test_clients=test_nodes,
    batch_size=32,
    dataset_name=dataset_name,
    num_workers=0,
    fed_dir=fed_dir,
    device="cuda",
    num_features=num_features,
    seed=42,
    surrogate=loaded_data_surrogate_DP["surrogate"],
)

# predictions with BB model

loaded_data_surrogate_DP["model_predictions"] = get_model_predictions(
    model=loaded_data_surrogate_DP["model"], X_test=X_test
)

loaded_data_surrogate_NO_DP["model_predictions"] = copy.copy(
    loaded_data_surrogate_DP["model_predictions"]
)

# Compute the base value

base_value = np.mean(X_test)
print(len(X_test))

# X_test = X_test[:10]
get_explanations(
    model_predictions=loaded_data_surrogate_DP["model_predictions"],
    X_test=X_test,
    loaded_data=loaded_data_surrogate_DP,
    base_value=base_value,
    base_path=base_path,
    explainer_privacy_levels=explainer_privacy_levels,
    store_path=store_path,
)


get_explanations(
    model_predictions=loaded_data_surrogate_NO_DP["model_predictions"],
    X_test=X_test,
    loaded_data=loaded_data_surrogate_NO_DP,
    base_value=base_value,
    base_path=base_path,
    explainer_privacy_levels=explainer_privacy_levels,
    store_path=store_path,
)

print(loaded_data_surrogate_DP["explanations_NO_DP"][0])
print(loaded_data_surrogate_NO_DP["explanations_NO_DP"][0])

print(loaded_data_surrogate_DP["explanations_NO_DP"][1])
print(loaded_data_surrogate_NO_DP["explanations_NO_DP"][1])

print(loaded_data_surrogate_DP["explanations_NO_DP"][2])
print(loaded_data_surrogate_NO_DP["explanations_NO_DP"][2])


plot_base_explainer_shap_values(
    loaded_data=loaded_data_surrogate_DP,
    explainer_privacy_levels=explainer_privacy_levels,
    X_test=X_test,
    feature_names=feature_names,
    title=f"Mean SHAP Values of Features for NON Private Model {dataset_name}",
    explainer_name="explanations_NO_DP",
)

if os.path.isfile(f"{metrics_folder}/metrics_surrogate_DP.pkl"):
    metrics_surrogate_DP = dill.load(
        open(
            f"{metrics_folder}/metrics_surrogate_DP.pkl",
            "rb",
        )
    )
else:
    metrics_surrogate_DP = compute_metrics(
        X_test,
        loaded_data_surrogate_DP,
        explainer_privacy_levels,
        top_k_private_bb,
        feature_names,
    )
    dill.dump(
        metrics_surrogate_DP,
        open(
            f"{metrics_folder}/metrics_surrogate_DP.pkl",
            "wb",
        ),
    )

if os.path.isfile(f"{metrics_folder}/metrics_surrogate_NO_DP.pkl"):
    metrics_surrogate_NO_DP = dill.load(
        open(
            f"{metrics_folder}/metrics_surrogate_NO_DP.pkl",
            "rb",
        )
    )
else:
    metrics_surrogate_NO_DP = compute_metrics(
        X_test,
        loaded_data_surrogate_NO_DP,
        explainer_privacy_levels,
        top_k_private_bb,
        feature_names,
    )
    dill.dump(
        metrics_surrogate_NO_DP,
        open(
            f"{metrics_folder}/metrics_surrogate_NO_DP.pkl",
            "wb",
        ),
    )


if os.path.isfile(f"{metrics_folder}/metrics_single_explainer_comparison.pkl"):
    metrics_single_explainer = dill.load(
        open(
            f"{metrics_folder}/metrics_single_explainer_comparison.pkl",
            "rb",
        )
    )
else:
    metrics_single_explainer = compute_metrics_single_explainer(
        X_test,
        loaded_data_surrogate_NO_DP,
        loaded_data_surrogate_DP,
        top_k_private_bb,
        feature_names,
    )
    dill.dump(
        metrics_single_explainer,
        open(
            f"{metrics_folder}/metrics_single_explainer_comparison.pkl",
            "wb",
        ),
    )
