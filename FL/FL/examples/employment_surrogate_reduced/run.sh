for i in $(seq 0 3);
do
    poetry run python /home/lcorbucci/private_fastshap/FL/FL/examples/employment_surrogate_reduced/../../main.py --node_shuffle_seed=$i --run_name Employment_Baseline_Surrogate --project_name EvalFastshap  --batch_size=2024 --epochs=10 --lr=0.09505717206283408 --optimizer=sgd --validation_batch_size=9456 --validation_samples=6 --dataset_name employment --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 50 --sampled_training_nodes 0.225 --sampled_validation_nodes 0 --sampled_test_nodes 1 --dataset_path /raid/lcorbucci/folktables/employment_data_reduced/ --seed 42 --wandb True --split_approach non_iid --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --split_approach non_iid --alpha_dirichlet 5 --train_surrogate True --bb_name ./DP_1_reduced.pth --splitted_data_dir federated_2
done


# DP 1
for i in $(seq 0 3);
do
    poetry run python /home/lcorbucci/private_fastshap/FL/FL/examples/employment_surrogate_reduced/../../main.py --node_shuffle_seed=$i --run_name Employment_DP_1_Surrogate --project_name EvalFastshap  --batch_size=3722 --clipping=4.330233187306873 --epochs=9 --lr=0.0026446736547189524 --optimizer=adam --validation_batch_size=1678 --validation_samples=7 --dataset_name employment --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 50 --sampled_training_nodes 0.225 --sampled_validation_nodes 0 --sampled_test_nodes 1 --dataset_path /raid/lcorbucci/folktables/employment_data_reduced/ --seed 42 --wandb True --split_approach non_iid --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --split_approach non_iid --alpha_dirichlet 5 --epsilon 1 --train_surrogate True --bb_name ./DP_1_reduced.pth --splitted_data_dir federated_2

done
