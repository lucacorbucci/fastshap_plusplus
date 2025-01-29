# NO DP
for i in $(seq 0 3);
do
    poetry run python /home/lcorbucci/private_fastshap/FL/FL/examples/income_surrogate_reduced/../../main.py --node_shuffle_seed=$i --run_name Income_Baseline_Surrogate --project_name EvalFastshap  --batch_size=2783 --epochs=6 --lr=0.0009635657780609708 --optimizer=adam --validation_batch_size=8801 --validation_samples=5 --dataset_name income --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 50 --sampled_training_nodes 0.225 --sampled_validation_nodes 0 --sampled_test_nodes 1 --dataset_path /raid/lcorbucci/folktables/income_data_reduced/ --seed 42 --wandb True --split_approach non_iid --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --split_approach non_iid --alpha_dirichlet 5 --train_surrogate True --bb_name ./DP_1_reduced.pth
done


# DP 1
for i in $(seq 0 3);
do
    poetry run python /home/lcorbucci/private_fastshap/FL/FL/examples/income_surrogate_reduced/../../main.py --batch_size=4080 --node_shuffle_seed=$i --run_name Income_DP_1_Surrogate --project_name EvalFastshap  --clipping=10.56665903042814 --epochs=8 --lr=0.004940058698261164 --optimizer=adam --validation_batch_size=2786 --validation_samples=9 --dataset_name income --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 50 --sampled_training_nodes 0.225 --sampled_validation_nodes 0 --sampled_test_nodes 1 --dataset_path /raid/lcorbucci/folktables/income_data_reduced/ --seed 42 --wandb True --split_approach non_iid --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --split_approach non_iid --alpha_dirichlet 5 --epsilon 1 --train_surrogate True --bb_name ./DP_1_reduced.pth
done