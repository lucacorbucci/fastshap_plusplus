
# # NO DP - Private Model
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=163 --eff_lambda=0.8159442690919287 --lr=0.018172668710876593 --num_samples=9 --optimizer=adam --paired_sampling=True --validation_samples=12 --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_DP_1.pt --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_NO_DP_private_model

# # EPS 1 - Private Model
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=121 --clipping=18.512962571196137 --eff_lambda=0.9778403174901984 --lr=0.0018304152088135855 --num_samples=4 --optimizer=adam --paired_sampling=False --validation_samples=17  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_DP_1.pt --epsilon 1 --dataset_name mnist --epochs 20 -save_model True --model_name mnist_explainer_private_model_DP_1

# # Eps 2 - Private Model
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=124 --clipping=18.05952393269368 --eff_lambda=0.7520130335033781 --lr=0.01678601130482112 --num_samples=9 --optimizer=adam --paired_sampling=True --validation_samples=11  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_DP_1.pt --epsilon 2 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_private_model_DP_2

# # EPS 3 - Private Model
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=253 --clipping=1.8360600539415912 --eff_lambda=0.8247738145434487 --lr=0.001480305577809597 --num_samples=2 --optimizer=adam --paired_sampling=True --validation_samples=6 --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_DP_1.pt --epsilon 3 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_private_model_DP_3

# # EPS 4 - Private Model
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=213 --clipping=17.5431306268173 --eff_lambda=0.542117562876822 --lr=0.005900710140033064 --num_samples=13 --optimizer=adam --paired_sampling=False --validation_samples=25 --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_DP_1.pt --epsilon 4 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_private_model_DP_4

# # EPS 5 - Private Model
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=250 --clipping=12.559954504054032 --eff_lambda=0.5098674813459774 --lr=0.009921259792728091 --num_samples=24 --optimizer=adam --paired_sampling=True --validation_samples=7  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_DP_1.pt --epsilon 5 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_private_model_DP_5

# # EPS 10 - Private Model
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=216 --clipping=6.558794673017595 --eff_lambda=0.27108287936340303 --lr=0.008335621774996567 --num_samples=8 --optimizer=adam --paired_sampling=True --validation_samples=22 --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_DP_1.pt --epsilon 10 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_private_model_DP_10
# # EPS 100 - Private Model
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=50 --clipping=10.6309596711324 --eff_lambda=0.8668495840655795 --lr=0.0023957189469397005 --num_samples=18 --optimizer=adam --paired_sampling=True --validation_samples=12 --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_DP_1.pt --epsilon 100 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_private_model_DP_100


# # DP 2 private
# poetry run wandb agent lucacorbucci/private-fastshap-mnist/oqu4gvmf --count 30

# # Dp 5 private
# poetry run wandb agent lucacorbucci/private-fastshap-mnist/725l9dmd  --count 30


# DP 4
poetry run wandb agent lucacorbucci/private-fastshap-mnist/5fzyboi0 --count 100

# dp 5
poetry run wandb agent lucacorbucci/private-fastshap-mnist/bqc04n5g --count 70

# DP 10
poetry run wandb agent lucacorbucci/private-fastshap-mnist/mht3cr35 --count 100

# DP 100
poetry run wandb agent lucacorbucci/private-fastshap-mnist/q9aum4o9 --count 100