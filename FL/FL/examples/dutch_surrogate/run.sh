# NO DP
for i in $(seq 0 3);
do
    poetry run python /home/lcorbucci/private_fastshap/FL/FL/examples/dutch_surrogate/../../main.py --node_shuffle_seed=$i --run_name Dutch_Baseline_Surrogate --project_name EvalFastshap  --batch_size=182 --epochs=10 --lr=0.0074656312927521085 --optimizer=adam --validation_batch_size=1220 --validation_samples=5 --dataset_name dutch --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 50 --sampled_training_nodes 0.15 --sampled_validation_nodes 0 --sampled_test_nodes 1 --dataset_path ../../../../data/dutch/ --seed 42 --wandb True --split_approach non_iid  --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --split_approach non_iid --alpha_dirichlet 5 --train_surrogate True --bb_name ./bb_NO_DP.pth # --save_aggregated_model True --aggregated_model_name surrogate_NO_DP   
done

# # DP 1
for i in $(seq 0 3);
do
    poetry run python /home/lcorbucci/private_fastshap/FL/FL/examples/dutch_surrogate/../../main.py --node_shuffle_seed=$i --run_name Dutch_DP_1_Surrogate --project_name EvalFastshap  --batch_size=258 --clipping=14.837537623481875 --epochs=6 --lr=0.005080286124536868 --optimizer=adam --validation_batch_size=3804 --validation_samples=6 --dataset_name dutch --fl_rounds 10 --num_client_cpus 1 --num_client_gpus 0.1 --tabular True --num_nodes 50 --sampled_training_nodes 0.15 --sampled_validation_nodes 0 --sampled_test_nodes 1 --dataset_path ../../../../data/dutch/ --seed 42 --wandb True --split_approach non_iid --fraction_fit_nodes 0.8 --fraction_validation_nodes 0 --fraction_test_nodes 0.2 --device cuda --cross_device True --split_approach non_iid --alpha_dirichlet 5 --epsilon 1 --train_surrogate True --bb_name ./bb_DP_1.pth # --save_aggregated_model True --aggregated_model_name surrogate_DP_1
done