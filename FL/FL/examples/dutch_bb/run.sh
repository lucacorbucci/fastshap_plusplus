# NO DP
for i in $(seq 0 3);
do
    poetry run python /home/lcorbucci/private_fastshap/FL/FL/examples/dutch_bb/../../main.py --node_shuffle_seed=$i --run_name Dutch_Baseline_Model --project_name EvalFastshap  --batch_size=249 --epochs=9 --lr=0.011676345366922853 --optimizer=adam --dataset_name dutch --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 50 --sampled_training_nodes 0.15 --sampled_validation_nodes 0 --sampled_test_nodes 1 --dataset_path ../../../../data/dutch/ --seed 42 --wandb True --split_approach non_iid  --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --split_approach non_iid --alpha_dirichlet 5 #--save_aggregated_model True
done


# DP 1
for i in $(seq 0 3);
do
    poetry run python /home/lcorbucci/private_fastshap/FL/FL/examples/dutch_bb/../../main.py --node_shuffle_seed=$i --run_name Dutch_DP_1_Model --project_name EvalFastshap --batch_size=294 --clipping=5.014101429939097 --epochs=10 --lr=0.04394945638200623 --optimizer=adam --dataset_name dutch --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 50 --sampled_training_nodes 0.15 --sampled_validation_nodes 0 --sampled_test_nodes 1 --dataset_path ../../../../data/dutch/ --seed 42 --wandb True --split_approach non_iid --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --split_approach non_iid --alpha_dirichlet 5 --epsilon 1 --save_aggregated_model True --aggregated_model_name bb_DP_1
done
