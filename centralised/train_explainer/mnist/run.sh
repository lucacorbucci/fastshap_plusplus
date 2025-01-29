# DP 1
poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=20 --clipping=2.6419926407273473 --eff_lambda=0.3842413047887828 --lr=0.003186981957996576 --num_samples=16 --validation_samples=43  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 1 --dataset_name mnist --paired_sampling True --epochs 20 --optimizer adam --save_model True --model_name mnist_explainer_DP_1

# DP 2 
poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=54 --clipping=8.820626285968643 --eff_lambda=0.6416320301633982 --lr=0.0007839733277629669 --num_samples=43 --validation_samples=19  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 2 --dataset_name mnist --paired_sampling True --epochs 20 --optimizer adam --save_model True --model_name mnist_explainer_DP_2

# DP 3
poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=235 --clipping=4.918199298351522 --eff_lambda=0.0389396954810689 --lr=0.002097368625817152 --num_samples=19 --validation_samples=18  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 3 --dataset_name mnist --paired_sampling True --epochs 20 --optimizer adam --save_model True --model_name mnist_explainer_DP_3

# DP 4
poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=251 --clipping=2.8118912057589505 --eff_lambda=0.5316795192590571 --lr=0.002611178303194104 --num_samples=37 --validation_samples=37  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 4 --dataset_name mnist --paired_sampling True --epochs 20 --optimizer adam --save_model True --model_name mnist_explainer_DP_4

# DP 5
poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=61 --clipping=12.947920412850287 --eff_lambda=0.5774458470176755 --lr=0.0021101589862646823 --num_samples=38 --validation_samples=45  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 5 --dataset_name mnist --paired_sampling True --epochs 20 --optimizer adam --save_model True --model_name mnist_explainer_DP_5