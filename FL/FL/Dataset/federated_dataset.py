import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from Dataset.tabular_dataset import TabularDataset


class FederatedDataset:
    @staticmethod
    def get_distribution(preferences):
        return np.random.dirichlet(
            preferences.num_nodes * [preferences.alpha_dirichlet], size=1
        )

    @staticmethod
    def create_federated_dataset(preferences, X, y, z, split_name, distribution=None):
        """

        Args:
        """

        if preferences.split_approach == "iid":
            # random shuffle the data in X, y, z
            combined_data = list(zip(X, y, z))
            random.shuffle(combined_data)
            X, y, z = zip(*combined_data)
            # split the data in num_nodes
            samples_per_node = len(y) // preferences.num_nodes
            nodes = []
            for i in range(preferences.num_nodes):
                nodes.append(
                    {
                        "x": np.array(
                            X[i * samples_per_node : (i + 1) * samples_per_node]
                        ),
                        "y": np.array(
                            y[i * samples_per_node : (i + 1) * samples_per_node]
                        ),
                        "z": np.array(
                            z[i * samples_per_node : (i + 1) * samples_per_node]
                        ),
                    }
                )
            # remove the old files in the data folder
            # if split_name == "train":
            #     os.system(f"rm -rf {preferences.dataset_path}/federated/*")
            for client_name, client in enumerate(nodes):
                # Append 1 to each samples
                custom_dataset = TabularDataset(
                    x=np.hstack(
                        (client["x"], np.ones((client["x"].shape[0], 1)))
                    ).astype(np.float32),
                    z=client["z"],  # .astype(np.float32),
                    y=client["y"],  # .astype(np.float32),
                )
                # Create the folder for the user client_name
                if not os.path.exists(
                    f"{preferences.dataset_path}/federated/{client_name}"
                ):
                    os.system(
                        f"mkdir {preferences.dataset_path}/federated/{client_name}"
                    )
                # store the dataset in the client folder with the name "train.pt"
                torch.save(
                    custom_dataset,
                    f"{preferences.dataset_path}/federated/{client_name}/{split_name}.pt",
                )

        elif preferences.split_approach == "non_iid":
            if distribution is None:
                raise ValueError(
                    "If you want to use the non_iid split you have to provide a distribution"
                )
            # non iid split with respect to the target
            # random shuffle the data in X, y, z
            combined_data = list(zip(X, y, z))
            random.shuffle(combined_data)
            X, y, z = zip(*combined_data)

            labels = np.array(list(y))

            idx = torch.tensor(list(range(len(labels))))

            index_per_label = {}
            for index, label in zip(idx, labels):
                if label.item() not in index_per_label:
                    index_per_label[label.item()] = []
                index_per_label[label.item()].append(index.item())

            # in list labels we have the labels of this dataset
            list_labels = {item.item() for item in labels}

            to_be_sampled = []
            # create the distribution for each class
            total_sum = 0
            for label in list_labels:
                # For each label we want a distribution over the num_partitions
                # we have to sample from the group of samples that have label equal
                # to label and not from the entire labels list.
                selected_labels = labels[labels == label]
                tmp_to_be_sampled = np.random.choice(
                    preferences.num_nodes, len(selected_labels), p=distribution[0]
                )
                total_sum += len(tmp_to_be_sampled)
                # Inside to_be_sampled we save a counter for each label
                # The counter is the number of samples that we want to sample for each
                # partition
                to_be_sampled.append(Counter(tmp_to_be_sampled))

            # create the partitions
            partitions_index = {
                f"node_{node_name}": [] for node_name in range(0, preferences.num_nodes)
            }
            for class_index, distribution_samples in zip(list_labels, to_be_sampled):
                for node_name, samples in distribution_samples.items():
                    partitions_index[f"node_{node_name}"] += index_per_label[
                        class_index
                    ][:samples]

                    index_per_label[class_index] = index_per_label[class_index][
                        samples:
                    ]

            total = 0
            for cluster, samples in partitions_index.items():
                total += len(samples)

            assert total == len(labels)

            partitions_labels = {
                node: np.array(y)[indexes] for node, indexes in partitions_index.items()
            }

            partitions_data = {
                node: np.array(X)[indexes] for node, indexes in partitions_index.items()
            }

            partitions_sensitive_values = {
                node: np.array(z)[indexes] for node, indexes in partitions_index.items()
            }

            nodes = []
            used_samples = 0
            for i in range(preferences.num_nodes):
                nodes.append(
                    {
                        "x": np.array(partitions_data[f"node_{i}"]),
                        "y": np.array(partitions_labels[f"node_{i}"]),
                        "z": np.array(partitions_sensitive_values[f"node_{i}"]),
                    }
                )
                assert len(np.array(partitions_data[f"node_{i}"])) == len(
                    np.array(partitions_labels[f"node_{i}"])
                )
                assert len(np.array(partitions_data[f"node_{i}"])) == len(
                    np.array(partitions_sensitive_values[f"node_{i}"])
                )
                used_samples += len(np.array(partitions_data[f"node_{i}"]))
            # remove the old files in the data folder
            # if split_name == "train":
            #     os.system(f"rm -rf {preferences.dataset_path}/federated/*")
            distributions = []
            for client_name, client in enumerate(nodes):
                # Append 1 to each samples
                custom_dataset = TabularDataset(
                    x=np.hstack(
                        (client["x"], np.ones((client["x"].shape[0], 1)))
                    ).astype(np.float32),
                    z=client["z"],  # .astype(np.float32),
                    y=client["y"],  # .astype(np.float32),
                )
                distributions.append(Counter(client["y"]))

                # Create the folder for the user client_name
                if not os.path.exists(
                    f"{preferences.dataset_path}/federated/{client_name}"
                ):
                    os.system(
                        f"mkdir {preferences.dataset_path}/federated/{client_name}"
                    )
                # store the dataset in the client folder with the name "train.pt"
                torch.save(
                    custom_dataset,
                    f"{preferences.dataset_path}/federated/{client_name}/{split_name}.pt",
                )

            # plot the distribution of the labels
            labels = ["Client " + str(i) for i in range(len(distributions))]
            zeros = [dist[0] for dist in distributions]
            ones = [dist[1] for dist in distributions]

            x = np.arange(len(labels))  # the label locations
            width = 0.35  # the width of the bars

            fig, ax = plt.subplots()
            fig.set_size_inches(20, 10)

            rects1 = ax.bar(x - width / 2, zeros, width, label="0")
            rects2 = ax.bar(x + width / 2, ones, width, label="1")

            ax.set_ylabel("Counts")
            ax.set_title("Distribution of 0 and 1 across clients")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            plt.xticks(rotation=90)
            ax.legend()

            fig.tight_layout()
            plt.savefig(f"./distribution.png")
        elif preferences.split_approach == "representative_diversity":
            number_unfair_nodes = int(
                preferences.num_nodes * preferences.ratio_unfair_nodes
            )
            number_fair_nodes = preferences.num_nodes - number_unfair_nodes
            nodes, remaining_data = FederatedDataset.representative_diversity_approach(
                X, y, z, preferences.num_nodes
            )
            sensitive_features = np.array([item.item() for item in z])
            labels = np.array([item for item in y])
            labels_and_sensitive = list(zip(labels, sensitive_features))

            fair_nodes, unfair_nodes = FederatedDataset.create_unfair_nodes(
                fair_nodes=nodes[:number_fair_nodes],
                nodes_to_unfair=nodes[number_fair_nodes:],
                remaining_data=remaining_data,
                group_to_reduce=preferences.group_to_reduce,
                group_to_increment=preferences.group_to_increment,
                ratio_unfairness=preferences.ratio_unfairness,
                # combination=labels_and_sensitive,
            )
            client_data = fair_nodes + unfair_nodes

            # transform client data so that they are compatiblw with the
            # other functions
            tmp_data = []
            for client in client_data:
                tmp_x = []
                tmp_y = []
                tmp_z = []
                for sample in client:
                    tmp_x.append(sample["x"])
                    tmp_y.append(sample["y"])
                    tmp_z.append(sample["z"])
                tmp_data.append(
                    {"x": np.array(tmp_x), "y": np.array(tmp_y), "z": np.array(tmp_z)}
                )
            client_data = tmp_data

            # remove the old files in the data folder
            if split_name == "train":
                os.system(f"rm -rf {preferences.dataset_path}/federated/*")
            for client_name, client in enumerate(client_data):
                # Append 1 to each samples
                print(client_name)
                custom_dataset = TabularDataset(
                    x=np.hstack(
                        (client["x"], np.ones((client["x"].shape[0], 1)))
                    ).astype(np.float32),
                    z=client["z"],  # .astype(np.float32),
                    y=client["y"],  # .astype(np.float32),
                )
                # Create the folder for the user client_name
                if not os.path.exists(
                    f"{preferences.dataset_path}/federated/{client_name}"
                ):
                    os.system(
                        f"mkdir {preferences.dataset_path}/federated/{client_name}"
                    )
                # store the dataset in the client folder with the name "train.pt"
                torch.save(
                    custom_dataset,
                    f"{preferences.dataset_path}/federated/{client_name}/{split_name}.pt",
                )

        else:
            raise ValueError(
                f"Split approach {preferences.split_approach} not supported."
            )

    @staticmethod
    def create_iid_split():
        pass

    def representative_diversity_approach(X, y, z, num_nodes):
        """
        With this approach we want to distribute the data among the nodes in a representative diversity way.
        This means that each node has the same ratio of each group that we are observing in the dataset

        params:
        X: numpy array of shape (N, D) where N is the number of samples and D is the number of features
        y: numpy array of shape (N, ) where N is the number of samples. Here we have the samples labels
        z: numpy array of shape (N, ) where N is the number of samples. Here we have the samples sensitive features
        num_nodes: number of nodes to generate
        number_of_samples_per_node: number of samples that we want in each node. Can be None, in this case we just use
            len(y)//num_nodes
        """
        number_of_samples_per_node = None
        samples_per_node = (
            number_of_samples_per_node
            if number_of_samples_per_node
            else len(y) // num_nodes
        )
        # create the nodes sampling from the dataset wihout replacement
        dataset = [{"x": x_, "y": y_, "z": z_} for x_, y_, z_ in zip(X, y, z)]
        # shuffle the dataset
        np.random.shuffle(dataset)

        # Distribute the data among the nodes with a random sample from the dataset
        # considering the number of samples per node
        nodes = []
        for i in range(num_nodes):
            nodes.append([])
            nodes[i].extend(dataset[:samples_per_node])
            dataset = dataset[samples_per_node:]

        # Create the dictionary with the remaining data
        remaining_data = {}
        for sample in dataset:
            if (sample["y"], sample["z"]) not in remaining_data:
                remaining_data[(sample["y"], sample["z"])] = []
            remaining_data[(sample["y"], sample["z"])].append(sample)

        return nodes, remaining_data

    @staticmethod
    def create_unfair_nodes(
        fair_nodes: list,
        nodes_to_unfair: list,
        remaining_data: dict,
        group_to_reduce: tuple,
        group_to_increment: tuple,
        ratio_unfairness: tuple,
    ):
        """
        This function creates the unfair nodes. It takes the nodes that we want to be unfair and the remaining data
        and it returns the unfair nodes created by reducing the group_to_reduce and incrementing the group_to_increment
        based on the ratio_unfairness

        params:
        nodes_to_unfair: list of nodes that we want to make unfair
        remaining_data: dictionary with the remaining data that we will use to replace the
            samples that we remove from the nodes_to_unfair
        group_to_reduce: the group that we want to be unfair. For instance, in the case of binary target and binary sensitive value
            we could have (0,0), (0,1), (1,0) or (1,1)
        group_to_increment: the group that we want to increment. For instance, in the case of binary target and binary sensitive value
            we could have (0,0), (0,1), (1,0) or (1,1)
        ratio_unfairness: tuple (min, max) where min is the minimum ratio of samples that we want to remove from the group_to_reduce
        """
        # assert (
        #     remaining_data[group_to_reduce] != []
        # ), "Choose a different group to be unfair"
        # remove the samples from the group that we want to be unfair
        unfair_nodes = []
        number_of_samples_to_add = []
        removed_samples = []

        for node in nodes_to_unfair:
            node_data = []
            count_sensitive_group_samples = 0
            # We count how many sample each node has from the group that we want to reduce
            for sample in node:
                if (sample["y"], sample["z"]) == group_to_reduce:
                    count_sensitive_group_samples += 1

            # We compute the number of samples that we want to remove from the group_to_reduce
            # based on the ratio_unfairness
            current_ratio = np.random.uniform(ratio_unfairness[0], ratio_unfairness[1])
            samples_to_be_removed = int(count_sensitive_group_samples * current_ratio)
            number_of_samples_to_add.append(samples_to_be_removed)

            for sample in node:
                # Now we remove the samples from the group_to_reduce
                # and we store them in removed_samples
                if (
                    sample["y"],
                    sample["z"],
                ) == group_to_reduce and samples_to_be_removed > 0:
                    samples_to_be_removed -= 1
                    removed_samples.append(sample)
                else:
                    node_data.append(sample)
            unfair_nodes.append(node_data)

        # Now we have to distribute the removed samples among the fair nodes
        max_samples_to_add = len(removed_samples) // len(fair_nodes)
        for node in fair_nodes:
            node.extend(removed_samples[:max_samples_to_add])
            removed_samples = removed_samples[max_samples_to_add:]

        if group_to_increment:
            # Now we have to remove the samples from the group_to_increment
            # from the fair_nodes based on the number_of_samples_to_add
            for node in fair_nodes:
                samples_to_remove = sum(number_of_samples_to_add) // len(fair_nodes)
                for index, sample in enumerate(node):
                    if (
                        sample["y"],
                        sample["z"],
                    ) == group_to_increment and samples_to_remove > 0:
                        if (sample["y"], sample["z"]) not in remaining_data:
                            remaining_data[group_to_increment] = []
                        remaining_data[group_to_increment].append(sample)
                        samples_to_remove -= 1
                        node.pop(index)
                if sum(number_of_samples_to_add) > 0:
                    assert samples_to_remove == 0, "Not enough samples to remove"
            # assert sum(number_of_samples_to_add) <= len(
            # remaining_data[group_to_increment]
            # ), "Too many samples to add"
            # now we have to add the same amount of data taken from group_to_unfair
            for node, samples_to_add in zip(unfair_nodes, number_of_samples_to_add):
                node.extend(remaining_data[group_to_increment][:samples_to_add])
                remaining_data[group_to_increment] = remaining_data[group_to_increment][
                    samples_to_add:
                ]

        return fair_nodes, unfair_nodes
