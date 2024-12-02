# NO DP
poetry run python /home/lcorbucci/private_fastshap/FL/FL/examples/dutch_surrogate/../../main.py --batch_size=182 --epochs=10 --lr=0.0074656312927521085 --optimizer=adam --validation_batch_size=1220 --validation_samples=5 --dataset_name dutch --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 50 --sampled_training_nodes 0.15 --sampled_validation_nodes 0 --sampled_test_nodes 1 --dataset_path ../../../../data/dutch/ --seed 42 --wandb True --split_approach non_iid --alpha_dirichlet 1 --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --split_approach non_iid --alpha_dirichlet 5 --train_surrogate True --bb_name ./bb_NO_DP.pth # --save_aggregated_model True

# # DP 1
# poetry run python /home/lcorbucci/private_fastshap/FL/FL/examples/dutch_surrogate/../../main.py --batch_size=501 --clipping=9.435864847237212 --epochs=9 --lr=0.004743623625061786 --optimizer=adam --validation_batch_size=6385 --validation_samples=5 --dataset_name dutch --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 50 --sampled_training_nodes 0.15 --sampled_validation_nodes 0 --sampled_test_nodes 1 --dataset_path ../../../../data/dutch/ --seed 42 --wandb True --split_approach non_iid --alpha_dirichlet 1 --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --split_approach non_iid --alpha_dirichlet 5 --epsilon 1 --train_surrogate True --bb_name ./bb_DP_1.pth --save_aggregated_model True