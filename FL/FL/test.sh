CUDA_VISIBLE_DEVICES=0 poetry run python main.py --dataset_name adult --dataset_path ../data/adult/ --epochs 1 --batch_size 64 --lr 0.001 --seed 0 --optimizer adam --tabular True --num_nodes 20 --split_approach iid --lr 0.1 --seed 42 --node_shuffle_seed 11 --validation_size 0.2 --sampled_training_nodes 0.5 --sampled_validation_nodes 1.0 --sampled_test_nodes 1.0 --num_client_cpus 1.0 --num_client_gpus 0.5 --fl_rounds 10 --save_local_models True --save_aggregated_model True --device cuda 