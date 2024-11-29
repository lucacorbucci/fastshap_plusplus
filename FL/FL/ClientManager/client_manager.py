# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower ClientManager."""

import os
import random
import threading
from logging import INFO
from typing import Dict, List, Optional

import dill
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from Utils.preferences import Preferences


class SimpleClientManager(ClientManager):
    """Provides a pool of available clients."""

    def __init__(
        self,
        preferences: Preferences = None,
    ) -> None:
        """Creates a SimpleClientManager.

        Parameters:
        -----------
        num_clients : int
            The pool size.
        sort_clients : bool
            If True, the clients will be sorted by their id.
        num_training_nodes : int
            The number of nodes to use for training.
        num_validation_nodes : int
            The number of nodes to use for validation.
        num_test_nodes : int
            The number of nodes to use for testing.
        node_shuffle_seed: int
            The seed to use for shuffling the nodes.
        seed: int
            The seed that we used to split the dataset
            and to assign nodes to the different sets.
        """
        self.seed = preferences.seed
        random.seed(self.seed)
        self.num_clients = preferences.num_nodes
        self.clients: Dict[str, ClientProxy] = {}
        self.clients_list: List[str] = []
        self.training_clients_list: List[str] = []
        self.validation_clients_list: List[str] = []
        self.test_clients_list: List[str] = []
        self._cv = threading.Condition()
        self.current_index_training = 0
        self.current_index_validation = 0
        self.current_index_test = 0

        # Percentage of nodes sampled during the training from
        # train, validation and test set
        self.sampled_training_nodes = preferences.sampled_training_nodes
        self.sampled_validation_nodes = preferences.sampled_validation_nodes
        self.sampled_test_nodes = preferences.sampled_test_nodes

        # Fraction of nodes for training, validation and test
        self.num_training_nodes = preferences.num_training_nodes
        self.num_validation_nodes = preferences.num_validation_nodes
        self.num_test_nodes = preferences.num_test_nodes

        self.node_shuffle_seed = preferences.node_shuffle_seed
        self.fed_dir = preferences.fed_dir
        self.ratio_unfair_nodes = (
            preferences.ratio_unfair_nodes if preferences.ratio_unfair_nodes else 0
        )
        self.fl_rounds = preferences.fl_rounds

        self.num_round_train = 0
        self.num_round_validation = 0
        self.num_round_test = 0
        self.preferences = preferences

    def __len__(self) -> int:
        return len(self.clients)

    def num_available(self, phase) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        if phase == "training":
            return len(self.training_clients_list)
        elif phase == "validation":
            return len(self.validation_clients_list)
        else:
            return len(self.test_clients_list)
        # return len(self)

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """Wait until at least `num_clients` are available.

        Blocks until the requested number of clients is available or until a
        timeout is reached. Current timeout default: 1 day.

        Parameters
        ----------
        num_clients : int
            The number of clients to wait for.
        timeout : int
            The time in seconds to wait for, defaults to 86400 (24h).

        Returns
        -------
        success : bool
        """
        with self._cv:
            return self._cv.wait_for(
                lambda: len(self.clients) >= num_clients, timeout=timeout
            )

    def pre_sample_clients_representative_diversity(
        self, fraction, ratio_unfair, unfair_group, fair_group, client_list
    ):
        sampled_nodes = {}
        for fl_round in range(self.fl_rounds):
            # number of nodes we have to select in each round
            nodes_to_sample = int(fraction * len(client_list))
            num_fair_nodes_sampled = int(nodes_to_sample * (1 - ratio_unfair))
            num_unfair_nodes_sampled = int(nodes_to_sample * ratio_unfair)
            start = fl_round * num_fair_nodes_sampled % len(fair_group)
            end = (fl_round * num_fair_nodes_sampled + num_fair_nodes_sampled) % len(
                fair_group
            )

            if start < end:
                fair_nodes_sampled = fair_group[start:end]
            else:
                fair_nodes_sampled = fair_group[start:] + fair_group[:end]

            if len(unfair_group) > 0:
                start = fl_round * num_unfair_nodes_sampled % len(unfair_group)
                end = (
                    fl_round * num_unfair_nodes_sampled + num_unfair_nodes_sampled
                ) % len(unfair_group)

                if start < end:
                    unfair_nodes_sampled = unfair_group[start:end]
                else:
                    unfair_nodes_sampled = unfair_group[start:] + unfair_group[:end]

                sampled_nodes[fl_round] = fair_nodes_sampled + unfair_nodes_sampled
            else:
                sampled_nodes[fl_round] = fair_nodes_sampled

        return sampled_nodes

    def pre_sample_clients(self, fraction, client_list):
        sampled_nodes = {}
        nodes_to_sample = int(fraction * len(client_list))
        print("Fraction: ", fraction, "Nodes to sample: ", nodes_to_sample)
        all_selected_nodes = []

        if nodes_to_sample == 0:
            return sampled_nodes, []

        for fl_round in range(self.fl_rounds):
            if fraction == 1.0:
                start = 0
                end = len(client_list)
            else:
                start = fl_round * nodes_to_sample % len(client_list)
                end = (fl_round * nodes_to_sample + nodes_to_sample) % len(client_list)

            if start < end:
                sampled_nodes[fl_round] = client_list[start:end]
            else:
                sampled_nodes[fl_round] = client_list[start:] + client_list[:end]
            all_selected_nodes += sampled_nodes[fl_round]

        return sampled_nodes, list(set(all_selected_nodes))

    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy

        Returns
        -------
        success : bool
            Indicating if registration was successful. False if ClientProxy is
            already registered or can not be registered for any reason.
        """
        if client.cid in self.clients:
            return False

        self.clients[client.cid] = client
        # Add the client to the list of available clients and then
        # shuffle the list
        self.clients_list.append(client.cid)

        if self.num_clients == len(self.clients_list):
            # Here we seed the random number generator with the
            # seed a different seed that the one that we use in other parts of the
            # code. This is because we want to be sure that the nodes that we select
            # for training, validation and testing changes when we change the seed.

            # In the case of a cross device experiment with a representative diversity
            # distribution
            if (
                self.preferences.cross_device
                and self.preferences.split_approach == "representative_diversity"
            ):
                # I want to be sure that in the test set we have always the same nodes
                # and that the distribution of the disparities of the nodes is the same
                # as the ones in the training set.
                random.seed(self.seed)
                self.clients_list = [
                    str(client_id)
                    for client_id in sorted(
                        [int(client_id) for client_id in self.clients_list]
                    )
                ]
                print("Clients list: ", self.clients_list)

                fair_group_size = int(
                    len(self.clients_list) * (1 - self.ratio_unfair_nodes)
                )
                unfair_group = self.clients_list[fair_group_size:]
                fair_group = self.clients_list[:fair_group_size]

                fair_test_nodes = int(
                    self.fraction_test_nodes * (1 - self.ratio_unfair_nodes)
                )
                unfair_test_nodes = int(
                    self.fraction_test_nodes * self.ratio_unfair_nodes
                )

                self.fair_test_clients = fair_group[:fair_test_nodes]
                self.unfair_test_clients = unfair_group[:unfair_test_nodes]
                self.test_clients_list = (
                    fair_group[:fair_test_nodes] + unfair_group[:unfair_test_nodes]
                )

                sampled_nodes_test = self.pre_sample_clients_representative_diversity(
                    fraction=self.sampled_test_nodes,
                    ratio_unfair=self.ratio_unfair_nodes,
                    unfair_group=self.unfair_test_clients,
                    fair_group=self.fair_test_clients,
                    client_list=self.test_clients_list,
                )

                with open(f"{self.fed_dir}/test_nodes.pkl", "wb") as f:
                    dill.dump(sampled_nodes_test, f)

                with open(f"{self.fed_dir}/all_test_nodes.pkl", "wb") as f:
                    dill.dump(self.test_clients_list, f)

                self.remaining_fair = fair_group[fair_test_nodes:]
                self.remaining_unfair = unfair_group[unfair_test_nodes:]

                print("Nodes in the test set: ", self.test_clients_list)
                print("Fair Test Nodes: ", len(self.fair_test_clients))
                print("Unfair Test Nodes: ", len(self.unfair_test_clients))

                random.seed(self.node_shuffle_seed)
                random.shuffle(self.remaining_fair)
                random.shuffle(self.remaining_unfair)

                fair_train_nodes = int(
                    self.num_training_nodes * (1 - self.ratio_unfair_nodes)
                )
                unfair_train_nodes = int(
                    self.num_training_nodes * self.ratio_unfair_nodes
                )
                self.fair_training_clients = self.remaining_fair[:fair_train_nodes]
                self.unfair_training_clients = self.remaining_unfair[
                    :unfair_train_nodes
                ]
                self.training_clients_list = (
                    self.remaining_fair[:fair_train_nodes]
                    + self.remaining_unfair[:unfair_train_nodes]
                )

                sampled_nodes_train = self.pre_sample_clients_representative_diversity(
                    fraction=self.fraction_train,
                    ratio_unfair=self.ratio_unfair_nodes,
                    unfair_group=self.unfair_training_clients,
                    fair_group=self.fair_training_clients,
                    client_list=self.training_clients_list,
                )
                with open(f"{self.fed_dir}/train_nodes.pkl", "wb") as f:
                    dill.dump(sampled_nodes_train, f)

                with open(f"{self.fed_dir}/all_train_nodes.pkl", "wb") as f:
                    dill.dump(self.training_clients_list, f)

                counter_sampling = {}
                for sample_list in sampled_nodes_train.values():
                    for node in sample_list:
                        if node not in counter_sampling:
                            counter_sampling[str(node)] = 0
                        counter_sampling[str(node)] += 1

                with open(f"{self.fed_dir}/counter_sampling.pkl", "wb") as f:
                    dill.dump(counter_sampling, f)

                print("Nodes in the training set: ", self.training_clients_list)
                print("Fair Training Nodes: ", len(self.fair_training_clients))
                print("Unfair Training Nodes: ", len(self.unfair_training_clients))

                self.fair_validation_clients = self.remaining_fair[fair_train_nodes:]
                self.unfair_validation_clients = self.remaining_unfair[
                    unfair_train_nodes:
                ]
                self.validation_clients_list = (
                    self.fair_validation_clients + self.unfair_validation_clients
                )
                # I want to be sure that in self.validation_clients_list we have an alternation of
                # fair and unfair nodes

                if self.fraction_validation > 0:
                    sampled_nodes_validation = (
                        self.pre_sample_clients_representative_diversity(
                            fraction=self.fraction_validation,
                            ratio_unfair=self.ratio_unfair_nodes,
                            unfair_group=self.unfair_validation_clients,
                            fair_group=self.fair_validation_clients,
                            client_list=self.validation_clients_list,
                        )
                    )
                    with open(f"{self.fed_dir}/validation_nodes.pkl", "wb") as f:
                        dill.dump(sampled_nodes_validation, f)

                    with open(f"{self.fed_dir}/all_validation_nodes.pkl", "wb") as f:
                        dill.dump(self.validation_clients_list, f)

                random.seed(self.seed)

                print("Nodes in the validation set: ", self.validation_clients_list)
                print("Fair Validation Nodes: ", len(self.fair_validation_clients))
                print("Unfair Validation Nodes: ", len(self.unfair_validation_clients))
                print(
                    "Total number of nodes: ",
                    len(self.test_clients_list)
                    + len(self.training_clients_list)
                    + len(self.validation_clients_list),
                )
            elif self.preferences.split_approach == "non_iid":
                random.seed(self.seed)
                random.shuffle(self.clients_list)

                test_nodes = self.clients_list[: self.num_test_nodes]
                self.clients_list = [
                    node for node in self.clients_list if node not in test_nodes
                ]
                print("Nodes in the test set: ", test_nodes)

                sampled_nodes_test, selected_test_nodes = self.pre_sample_clients(
                    fraction=self.sampled_test_nodes,
                    client_list=test_nodes,
                )

                with open(f"{self.fed_dir}/test_nodes.pkl", "wb") as f:
                    dill.dump(sampled_nodes_test, f)

                # In this case I'm in the cross-silo case
                # This means that each node has training, validation and test data
                # so each node could be used for training, validation and testing
                if self.sampled_validation_nodes:
                    print("Seeding with ", self.node_shuffle_seed)
                    random.seed(self.node_shuffle_seed)
                    random.shuffle(self.clients_list)
                    validation_nodes = self.clients_list[: self.num_validation_nodes]
                    self.clients_list = [
                        node
                        for node in self.clients_list
                        if node not in validation_nodes
                    ]

                    print("Nodes in the validation set: ", validation_nodes)
                    sampled_nodes_validation, selected_val_nodes = (
                        self.pre_sample_clients(
                            fraction=self.sampled_validation_nodes,
                            client_list=validation_nodes,
                        )
                    )
                    with open(f"{self.fed_dir}/validation_nodes.pkl", "wb") as f:
                        dill.dump(sampled_nodes_validation, f)
                else:
                    selected_val_nodes = []
                    sampled_nodes_validation = {}
                random.seed(self.seed)
                print("Nodes in the train set: ", self.clients_list)

                train_nodes = self.clients_list[: self.num_training_nodes]
                sampled_nodes_train, selected_train_nodes = self.pre_sample_clients(
                    fraction=self.sampled_training_nodes,
                    client_list=train_nodes,
                )
                with open(f"{self.fed_dir}/train_nodes.pkl", "wb") as f:
                    dill.dump(sampled_nodes_train, f)

                print("Nodes in the training set: ", selected_train_nodes)
                print("Sampled nodes for training: ", sampled_nodes_train)
                print("Nodes in the validation set: ", selected_val_nodes)
                print("Sampled nodes for validation: ", sampled_nodes_validation)
                print("Nodes in the test set: ", selected_test_nodes)
                print("Sampled nodes for test: ", sampled_nodes_test)
                counter_sampling = {}
                for sample_list in sampled_nodes_train.values():
                    for node in sample_list:
                        if node not in counter_sampling:
                            counter_sampling[str(node)] = 0
                        counter_sampling[str(node)] += 1

                with open(f"{self.fed_dir}/counter_sampling.pkl", "wb") as f:
                    dill.dump(counter_sampling, f)

                print("Counter Sampling: ", counter_sampling)
                random.seed(self.seed)

            else:
                raise NotImplementedError()

        with self._cv:
            self._cv.notify_all()

        return True

    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance.

        This method is idempotent.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy
        """
        if client.cid in self.clients:
            del self.clients[client.cid]
            self.clients_list.remove(client.cid)

            with self._cv:
                self._cv.notify_all()

    def all(self) -> Dict[str, ClientProxy]:
        """Return all available clients."""
        return self.clients

    def sample(
        self,
        num_clients: int,
        phase: str,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion

        if phase == "training":
            with open(f"{self.fed_dir}/train_nodes.pkl", "rb") as f:
                train_nodes = dill.load(f)

            sampled_clients = [
                self.clients[str(node)] for node in train_nodes[self.num_round_train]
            ]
            self.num_round_train += 1

            print(
                "===>>>> Sampled for training: ",
                [client.cid for client in sampled_clients],
            )
        elif phase == "validation":
            with open(f"{self.fed_dir}/validation_nodes.pkl", "rb") as f:
                validation_nodes = dill.load(f)

            sampled_clients = [
                self.clients[str(node)]
                for node in validation_nodes[self.num_round_validation]
            ]
            self.num_round_validation += 1
            print(
                "===>>>> Sampled for validation: ",
                [client.cid for client in sampled_clients],
            )
        else:
            with open(f"{self.fed_dir}/test_nodes.pkl", "rb") as f:
                test_nodes = dill.load(f)

            sampled_clients = [
                self.clients[str(node)] for node in test_nodes[self.num_round_test]
            ]
            self.num_round_test += 1

            print(
                "===>>>> Sampled for test: ", [client.cid for client in sampled_clients]
            )
        return sampled_clients
