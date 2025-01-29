import itertools

# Weakness 1.2
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr


# convergence check only
def compare_attr(attrA, attrB):
    """
    in:
    attrA: feature attributions, [n_points, n_features]
    attrB: feature attributions, [n_points, n_features]
    out:
    L2: L2 distance between 2 sets of feature attributions, [n_points]
    """
    L2 = np.linalg.norm(attrA - attrB, axis=1)
    return L2


def top_features(E, k):
    return np.argsort(E)[::-1][:k]


def rank(E, s):
    return np.where(np.argsort(E)[::-1] == s)[0][0]


def modified_rank_agreement(Ea, Eb, k):
    N, f = Ea.shape
    agreements = np.zeros(N)

    for i in range(N):
        top_features_Ea = top_features(Ea[i], k)
        top_features_Eb = top_features(Eb[i], k)

        agreement = 0
        for s in np.union1d(top_features_Ea, top_features_Eb):
            if s in top_features_Ea and s in top_features_Eb:
                rank_diff = abs(rank(Ea[i], s) - rank(Eb[i], s))
                weight = 1 - (rank_diff / k)
                agreement += weight

        agreements[i] = agreement / k

    return agreements


def pairwise_rank_agreement(e1, e2, k_values):
    n_features = len(e1)
    feature_indices = np.arange(n_features)

    agreements = {}

    for k in k_values:
        top_k_indices_e1 = np.argsort(e1)[-k:]
        top_k_indices_e2 = np.argsort(e2)[-k:]

        top_k_features = np.union1d(top_k_indices_e1, top_k_indices_e2)

        n_pairs = len(list(combinations(top_k_features, 2)))

        agreement_count = 0

        for feature_pair in combinations(top_k_features, 2):
            feature_i, feature_j = feature_pair

            if (e1[feature_i] > e1[feature_j] and e2[feature_i] > e2[feature_j]) or (
                e1[feature_i] < e1[feature_j] and e2[feature_i] < e2[feature_j]
            ):
                agreement_count += 1

        agreement = agreement_count / n_pairs
        agreements[k] = agreement

    return agreements


def rank_correlation_top_k(e1, e2, k_values):
    n_features = len(e1)
    feature_indices = np.arange(n_features)

    correlations = {}

    for k in k_values:
        top_k_indices_e1 = np.argsort(e1)[-k:]
        top_k_indices_e2 = np.argsort(e2)[-k:]

        top_k_features = np.union1d(top_k_indices_e1, top_k_indices_e2)

        top_k_values_e1 = e1[top_k_features]
        top_k_values_e2 = e2[top_k_features]

        rank_e1 = np.argsort(np.argsort(top_k_values_e1))
        rank_e2 = np.argsort(np.argsort(top_k_values_e2))

        correlation, _ = spearmanr(rank_e1, rank_e2)
        correlations[k] = correlation

    return correlations


# for convergence check
def compare_feature_ranks(attrA, attrB):
    """
    in:
    attrA: feature attributions, [n_points, n_features]
    attrB: feature attributions, [n_points, n_features]
    out:
    L2: L2 distance between 2 sets of feature ranks, [n_points]
    """
    ranksA = np.argsort(-np.abs(attrA), axis=1)
    ranksB = np.argsort(-np.abs(attrB), axis=1)
    L2 = np.linalg.norm(ranksA - ranksB, axis=1)
    return L2


def rankcorr(attrA, attrB):
    """
    in:
    attrA: feature attributions, [n_points, n_features]
    attrB: feature attributions, [n_points, n_features]
    out:
    corrs: rank correlation of attrA and attrB for each point, [n_points]
    """
    corrs = []
    # rank features (accounting for ties)
    all_feat_ranksA = rankdata(
        -np.abs(attrA), method="dense", axis=1
    )  # rankdata gives rank1 for smallest # --> we want rank1 for largest # (aka # with largest magnitude)
    all_feat_ranksB = rankdata(-np.abs(attrB), method="dense", axis=1)
    for row in range(attrA.shape[0]):
        # calculate correlation on ranks (iterate through rows: https://stackoverflow.com/questions/44947030/how-to-get-scipy-stats-spearmanra-b-compute-correlation-only-between-variable)
        rho, _ = pearsonr(all_feat_ranksA[row, :], all_feat_ranksB[row, :])
        corrs.append(rho)
    corrs = np.array(corrs)
    return corrs


def pairwise_rank_agreement(attrA, attrB):
    """
    inputs
    attrA: feature attributions, [n_points, n_features]
    attrB: feature attributions, [n_points, n_features]

    outputs:
    pairwise_distr: 1D numpy array (dimensions=(n_points,)) of pairwise rank agreement for each data point
    """
    n_points = attrA.shape[0]
    n_feat = attrA.shape[1]

    # rank of all features --> manually calculate rankings (instead of using 0, 1, ..., k ranking based on argsort output) to account for ties
    all_feat_ranksA = rankdata(
        -np.abs(attrA), method="dense", axis=1
    )  # rankdata gives rank1 for smallest # --> we want rank1 for largest # (aka # with largest magnitude)
    all_feat_ranksB = rankdata(-np.abs(attrB), method="dense", axis=1)

    # count # of pairs of features with same relative ranking
    feat_pairs_w_same_rel_rankings = np.zeros(n_points)

    # for each pair of features...
    for feat1, feat2 in itertools.combinations_with_replacement(range(n_feat), 2):
        if feat1 != feat2:
            rel_rankingA = (
                all_feat_ranksA[:, feat1] < all_feat_ranksA[:, feat2]
            )  # is rank(feat1) < rank(feat2) in attrA?
            rel_rankingB = (
                all_feat_ranksB[:, feat1] < all_feat_ranksB[:, feat2]
            )  # is rank(feat1) < rank(feat2) in attrB?
            feat_pairs_w_same_rel_rankings += (
                rel_rankingA == rel_rankingB
            )  # +1 if feat1 and feat2 have the same relative rankings in attrA and attrB

    pairwise_distr = feat_pairs_w_same_rel_rankings / comb(n_feat, 2)

    return pairwise_distr


def agreement_fraction(
    attrA, attrB, k, metric=["feature", "rank", "sign", "signedrank"]
):
    """
    inputs
    attrA: feature attributions, [n_points, n_features]
    attrB: feature attributions, [n_points, n_features]
    k: number of topk features to consider, integer
    metric: metric name, string

    outputs:
    metric_distr: 1D numpy array (dimensions=(n_points,)) of metric for each data point
    """

    # id of top-k features
    topk_idA = np.argsort(-np.abs(attrA), axis=1)[:, 0:k]
    topk_idB = np.argsort(-np.abs(attrB), axis=1)[:, 0:k]

    # rank of top-k features --> manually calculate rankings (instead of using 0, 1, ..., k ranking based on argsort output) to account for ties
    all_feat_ranksA = rankdata(
        -np.abs(attrA), method="dense", axis=1
    )  # rankdata gives rank1 for smallest # --> we want rank1 for largest # (aka # with largest magnitude)
    all_feat_ranksB = rankdata(-np.abs(attrB), method="dense", axis=1)
    topk_ranksA = np.take_along_axis(all_feat_ranksA, topk_idA, axis=1)
    topk_ranksB = np.take_along_axis(all_feat_ranksB, topk_idB, axis=1)

    # sign of top-k features
    topk_signA = np.take_along_axis(np.sign(attrA), topk_idA, axis=1)  # pos=1; neg=-1
    topk_signB = np.take_along_axis(np.sign(attrB), topk_idB, axis=1)

    # feature agreement = (# topk features in common)/k
    if metric == "feature":
        topk_setsA = [set(row) for row in topk_idA]
        topk_setsB = [set(row) for row in topk_idB]
        # check if: same id
        metric_distr = np.array(
            [
                len(setA.intersection(setB)) / k
                for setA, setB in zip(topk_setsA, topk_setsB)
            ]
        )

    # rank agreement
    elif metric == "rank":
        topk_idA_df = pd.DataFrame(topk_idA).applymap(str)  # id
        topk_idB_df = pd.DataFrame(topk_idB).applymap(str)
        topk_ranksA_df = pd.DataFrame(topk_ranksA).applymap(
            str
        )  # rank (accounting for ties)
        topk_ranksB_df = pd.DataFrame(topk_ranksB).applymap(str)
        # check if: same id + rank
        topk_id_ranksA_df = ("feat" + topk_idA_df) + ("rank" + topk_ranksA_df)
        topk_id_ranksB_df = ("feat" + topk_idB_df) + ("rank" + topk_ranksB_df)
        metric_distr = (topk_id_ranksA_df == topk_id_ranksB_df).sum(
            axis=1
        ).to_numpy() / k

    # sign agreement
    elif metric == "sign":
        topk_idA_df = pd.DataFrame(topk_idA).applymap(
            str
        )  # id (contains rank info --> order of features in columns)
        topk_idB_df = pd.DataFrame(topk_idB).applymap(str)
        topk_signA_df = pd.DataFrame(topk_signA).applymap(str)  # sign
        topk_signB_df = pd.DataFrame(topk_signB).applymap(str)
        # check if: same id + sign
        topk_id_signA_df = ("feat" + topk_idA_df) + (
            "sign" + topk_signA_df
        )  # id + sign (contains rank info --> order of features in columns)
        topk_id_signB_df = ("feat" + topk_idB_df) + ("sign" + topk_signB_df)
        topk_id_signA_sets = [
            set(row) for row in topk_id_signA_df.to_numpy()
        ]  # id + sign (remove order info --> by converting to sets)
        topk_id_signB_sets = [set(row) for row in topk_id_signB_df.to_numpy()]
        metric_distr = np.array(
            [
                len(setA.intersection(setB)) / k
                for setA, setB in zip(topk_id_signA_sets, topk_id_signB_sets)
            ]
        )

    # rank and sign agreement
    elif metric == "signedrank":
        topk_idA_df = pd.DataFrame(topk_idA).applymap(str)  # id
        topk_idB_df = pd.DataFrame(topk_idB).applymap(str)
        topk_ranksA_df = pd.DataFrame(topk_ranksA).applymap(
            str
        )  # rank (accounting for ties)
        topk_ranksB_df = pd.DataFrame(topk_ranksB).applymap(str)
        topk_signA_df = pd.DataFrame(topk_signA).applymap(str)  # sign
        topk_signB_df = pd.DataFrame(topk_signB).applymap(str)
        # check if: same id + rank + sign
        topk_id_ranks_signA_df = (
            ("feat" + topk_idA_df)
            + ("rank" + topk_ranksA_df)
            + ("sign" + topk_signA_df)
        )
        topk_id_ranks_signB_df = (
            ("feat" + topk_idB_df)
            + ("rank" + topk_ranksB_df)
            + ("sign" + topk_signB_df)
        )
        metric_distr = (topk_id_ranks_signA_df == topk_id_ranks_signB_df).sum(
            axis=1
        ).to_numpy() / k

    return metric_distr
