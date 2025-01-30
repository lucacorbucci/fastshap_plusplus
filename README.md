# Fastshap++ test

How to use Fastshap++ with Differential Privacy

To execute the code and reproduce the results of the paper, follow the instructions below:

- We managed the Python Dependencies using Poetry, so you need to install it first if you do not have it. You can find the instructions [here](https://python-poetry.org/docs/#installation).
- Once Poetry is installed, you can install the dependencies by running the following command in the root directory of the project:
```bash
poetry install
```

# Reproducing the results

In this section, we will show how to reproduce the results of the paper.

## Datasets

As shown in the paper, we considered three different datasets for our experiments: Dutch, ACS Income and ACS Employment. 
You can find in the ./dataset folder three subfolders, each one containing the datasets used in the paper:

- Dutch: for this dataset you'll find the file dutch_census.csv containing the Dutch Census data. When running experiments with this dataset, our code will automatically split the data into a predefined number of clients. You'll need to have a "federated" folder in the ./dataset/dutch folder.
- ACS Income: In this case, the data is already splitted into 50 clients and the code will just load the data from the files. You'll find the files in the ./dataset/acs_income folder, divided into 50 folders, each one containing the data for a client.
- ACS Employment: In this case, the data is already splitted into 50 clients and the code will just load the data from the files. You'll find the files in the ./dataset/acs_income folder, divided into 50 folders, each one containing the data for a client.

The datasets are already preprocessed and ready to be used following the preprocessing steps described in the paper. 


## Train a Black Box Model
The following commands show how to train a black box model with and without DP for the three datasets considered in the paper. All the commands assumes that you are in the ./FL/FL folder and that you installed and configured [wandb](https://wandb.ai/site) to log the results.

**Dutch**: To train a black box model with the Dutch dataset  with $\epsilon = 1$
```bash
    poetry run python main.py --node_shuffle_seed=1 --run_name Dutch_DP_1_Model --project_name EvalFastshap --batch_size=294 --clipping=5.014101429939097 --epochs=10 --lr=0.04394945638200623 --optimizer=adam --dataset_name dutch --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 50 --sampled_training_nodes 0.15 --sampled_validation_nodes 0 --sampled_test_nodes 1 --dataset_path ../../datasets/dutch/ --seed 42 --wandb True --split_approach non_iid --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --split_approach non_iid --alpha_dirichlet 5 --epsilon 1 --save_aggregated_model True --aggregated_model_name bb_DP_1
```

**ACS Income**:
```bash
    poetry run python main.py --node_shuffle_seed=1 --run_name Income_DP_1_Model --project_name EvalFastshap --batch_size=198 --clipping=5.10293483721213 --epochs=7 --lr=0.09797360858251336 --optimizer=adam --dataset_name income --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 51 --sampled_training_nodes 0.175 --sampled_validation_nodes 0 --sampled_test_nodes 0 --dataset_path /raid/lcorbucci/folktables/income_data_reduced/ --seed 42 --wandb True --split_approach non_iid --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --epsilon 1 --splitted_data_dir federated --save_aggregated_model True --aggregated_model_name DP_1_reduced
```


**ACS Employment**:

```bash
    poetry run python main.py --node_shuffle_seed=1 --run_name Employment_DP_1_Model --project_name EvalFastshap --batch_size=8442 --clipping=10.395438506476795 --epochs=9 --lr=0.07645692287323336 --optimizer=adam --dataset_name employment --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 51 --sampled_training_nodes 0.175 --sampled_validation_nodes 0 --sampled_test_nodes 1 --dataset_path /raid/lcorbucci/folktables/employment_data_reduced/ --seed 42 --wandb True --split_approach non_iid --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --epsilon 1 --splitted_data_dir federated_2 --save_aggregated_model True --aggregated_model_name DP_1_reduced
```

## Train a Surrogate Model

**Dutch**: To train a black box model with the Dutch dataset without DP, you can run the following command:

```bash
    poetry run python main.py --node_shuffle_seed=1 --run_name Dutch_DP_1_Surrogate --project_name EvalFastshap  --batch_size=258 --clipping=14.837537623481875 --epochs=6 --lr=0.005080286124536868 --optimizer=adam --validation_batch_size=3804 --validation_samples=6 --dataset_name dutch --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 50 --sampled_training_nodes 0.15 --sampled_validation_nodes 0 --sampled_test_nodes 1 --dataset_path ../../../../data/dutch/ --seed 42 --wandb True --split_approach non_iid --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --split_approach non_iid --alpha_dirichlet 5 --epsilon 1 --train_surrogate True --bb_name ./bb_DP_1.pth # --save_aggregated_model True --aggregated_model_name surrogate_DP_1
```


**ACS Income**:
```bash
        poetry run python main.py --node_shuffle_seed=1 --run_name Income_DP_1_Model --project_name EvalFastshap --batch_size=198 --clipping=5.10293483721213 --epochs=7 --lr=0.09797360858251336 --optimizer=adam --dataset_name income --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 51 --sampled_training_nodes 0.175 --sampled_validation_nodes 0 --sampled_test_nodes 0 --dataset_path /raid/lcorbucci/folktables/income_data_reduced/ --seed 42 --wandb True --split_approach non_iid --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --epsilon 1 --splitted_data_dir federated --save_aggregated_model True --aggregated_model_name DP_1_reduced
```

**ACS Employment**:
```bash
    poetry run python main.py --node_shuffle_seed=1 --run_name Employment_DP_1_Model --project_name EvalFastshap --batch_size=8442 --clipping=10.395438506476795 --epochs=9 --lr=0.07645692287323336 --optimizer=adam --dataset_name employment --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 51 --sampled_training_nodes 0.175 --sampled_validation_nodes 0 --sampled_test_nodes 1 --dataset_path /raid/lcorbucci/folktables/employment_data_reduced/ --seed 42 --wandb True --split_approach non_iid --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --epsilon 1 --splitted_data_dir federated_2 --save_aggregated_model True --aggregated_model_name DP_1_reduced
```

## Train an Explainer Model


**Dutch**: To train a black box model with the Dutch dataset without DP, you can run the following command:

If you want to train the black box with DP with $\epsilon = 1$

```bash
poetry run python main.py --batch_size=167 --clipping=3.045301219922139 --eff_lambda=0.5144162199810652 --epochs=6 --lr=0.0031527634744994227 --optimizer=adam --paired_sampling=False --validation_batch_size=9948 --validation_samples=9 --dataset_name dutch --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 50 --sampled_training_nodes 0.225 --sampled_validation_nodes 0 --sampled_test_nodes 1 --dataset_path ../../../../data/dutch/ --seed 42 --wandb True --split_approach non_iid --fraction_fit_nodes 0.6 --fraction_validation_nodes 0.8 --fraction_test_nodes 0.2 --fraction_validation_nodes 0 --device cuda --cross_device True --split_approach non_iid --alpha_dirichlet 5 --train_explainer True --surrogate_name ./dutch_surrogate_DP_1.pth --epsilon 1 --aggregated_model_name DP_1_DP_BB --save_aggregated_model True
```

**ACS Income**:

```bash
poetry run python /main.py --batch_size=304 --clipping=2.44245202440788 --eff_lambda=0.016843245445491983 --epochs=9 --lr=0.01471898586657231 --num_samples=18 --optimizer=sgd --paired_sampling=False --validation_samples=18 --dataset_name income --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 50 --sampled_training_nodes 0.225 --sampled_validation_nodes 0 --sampled_test_nodes 1 --dataset_path /raid/lcorbucci/folktables/income_data_reduced --seed 42 --wandb True --split_approach non_iid --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --split_approach non_iid --alpha_dirichlet 5 --train_explainer True --surrogate_name ./surrogate_DP_1_reduced.pth --epsilon 1 --splitted_data_dir federated_2 --save_aggregated_model True --aggregated_model_name explainer_DP_1
```

**ACS Employment**:

```bash
poetry run python main.py --batch_size=283 --clipping=2.586166101360625 --eff_lambda=0.16375400631673576 --epochs=10 --lr=0.03978244119280488 --num_samples=29 --optimizer=sgd --paired_sampling=False --validation_samples=43 --dataset_name employment --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 50 --sampled_training_nodes 0.225 --sampled_validation_nodes 0 --sampled_test_nodes 1 --dataset_path /raid/lcorbucci/folktables/employment_data_reduced/ --seed 42 --wandb True --split_approach non_iid --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --split_approach non_iid --alpha_dirichlet 5 --train_explainer True --surrogate_name ./surrogate_DP_1.pth --epsilon 1 --splitted_data_dir federated_2 --save_aggregated_model True --aggregated_model_name explainer_DP_1
```
