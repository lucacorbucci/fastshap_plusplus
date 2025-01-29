# bb_NO_DP_employment
for i in $(seq 0 3);
do
    poetry run python /home/lcorbucci/private_fastshap/FL/FL/examples/employment_bb_reduced/../../main.py --node_shuffle_seed=$i --run_name Employment_Baseline_Model --project_name EvalFastshap --batch_size=9974 --epochs=10 --lr=0.016325370178743515 --optimizer=adam --dataset_name employment --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 51 --sampled_training_nodes 0.175 --sampled_validation_nodes 0 --sampled_test_nodes 1 --dataset_path /raid/lcorbucci/folktables/employment_data_reduced/ --seed 42 --wandb True --split_approach non_iid --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --splitted_data_dir federated  --save_aggregated_model True --aggregated_model_name NO_DP_reduced
done


# bb_DP_1_employment
for i in $(seq 0 3);
do
    poetry run python /home/lcorbucci/private_fastshap/FL/FL/examples/employment_bb_reduced/../../main.py --node_shuffle_seed=$i --run_name Employment_DP_1_Model --project_name EvalFastshap --batch_size=8442 --clipping=10.395438506476795 --epochs=9 --lr=0.07645692287323336 --optimizer=adam --dataset_name employment --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 51 --sampled_training_nodes 0.175 --sampled_validation_nodes 0 --sampled_test_nodes 1 --dataset_path /raid/lcorbucci/folktables/employment_data_reduced/ --seed 42 --wandb True --split_approach non_iid --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --epsilon 1 --splitted_data_dir federated_2 --save_aggregated_model True --aggregated_model_name DP_1_reduced
done
