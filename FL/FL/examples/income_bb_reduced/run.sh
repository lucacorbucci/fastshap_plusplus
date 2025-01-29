# NO DP
for i in $(seq 0 3);
do
    poetry run python /home/lcorbucci/private_fastshap/FL/FL/examples/income_bb_reduced/../../main.py --node_shuffle_seed=$i --run_name Income_Baseline_Model --project_name EvalFastshap --batch_size=130 --epochs=10 --lr=0.03377082453146142 --optimizer=sgd --dataset_name income --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 51 --sampled_training_nodes 0.175 --sampled_validation_nodes 0 --sampled_test_nodes 1 --dataset_path /raid/lcorbucci/folktables/income_data_reduced/ --seed 42 --wandb True --split_approach non_iid --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --splitted_data_dir federated
done


# DP 1
for i in $(seq 0 3);
do
    poetry run python /home/lcorbucci/private_fastshap/FL/FL/examples/income_bb_reduced/../../main.py --node_shuffle_seed=$i --run_name Income_DP_1_Model --project_name EvalFastshap --batch_size=198 --clipping=5.10293483721213 --epochs=7 --lr=0.09797360858251336 --optimizer=adam --dataset_name income --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 51 --sampled_training_nodes 0.175 --sampled_validation_nodes 0 --sampled_test_nodes 0 --dataset_path /raid/lcorbucci/folktables/income_data_reduced/ --seed 42 --wandb True --split_approach non_iid --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --epsilon 1 --splitted_data_dir federated --save_aggregated_model True --aggregated_model_name DP_1_reduced
done
