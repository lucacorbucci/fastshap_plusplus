# # NO DP
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=236 --eff_lambda=0.5354794915050585 --lr=0.023339074206445504 --num_samples=23 --optimizer=adam --paired_sampling=True --validation_samples=8  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_NO_DP

# # EPS 1
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=228 --clipping=8.934576946572783 --eff_lambda=0.06525362270465063 --lr=0.009424192981584246 --num_samples=15 --optimizer=adam --paired_sampling=True --validation_samples=25  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 1 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_1

# # Eps 2
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=140 --clipping=6.882668403296307 --eff_lambda=0.8971269791431516 --lr=0.004492215834756371 --num_samples=4 --optimizer=adam --paired_sampling=False --validation_samples=6  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 2 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_2

# # EPS 3 -> rifare
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=152 --clipping=13.780841565351952 --eff_lambda=0.9194134891284488 --lr=0.005979772881628436 --num_samples=7 --optimizer=adam --paired_sampling=False --validation_samples=2  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 3 --dataset_name mnist --epochs 20 --epochs 20 --save_model True --model_name mnist_explainer_DP_3
# # EPS 4
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=140 --clipping=12.127429620256123 --eff_lambda=0.7594338000737504 --lr=0.001875081921968342 --num_samples=8 --optimizer=adam --paired_sampling=False --validation_samples=10  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 4 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_4

# # EPS 5
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=147 --clipping=13.062052707115836 --eff_lambda=0.13453365550065033 --lr=0.004640011642237676 --num_samples=13 --optimizer=adam --paired_sampling=False --validation_samples=11  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 5 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_5

# # EPS 10
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=76 --clipping=12.015045264363764 --eff_lambda=0.9930200681279556 --lr=0.006562881697186505 --num_samples=16 --optimizer=adam --paired_sampling=True --validation_samples=15  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 10 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_10

# # EPS 100
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=248 --clipping=1.006405059663407 --eff_lambda=0.30054759129788144 --lr=0.004963361488282572 --num_samples=22 --optimizer=adam --paired_sampling=False --validation_samples=9  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 100 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_100

# # DP 5 no private
# poetry run wandb agent lucacorbucci/private-fastshap-mnist/bqc04n5g --count 30

# # DP 1 private
# poetry run wandb agent lucacorbucci/private-fastshap-mnist/vfsvl5lo --count 30


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


# # no dp
# poetry run wandb agent lucacorbucci/private-fastshap-mnist/na7j5mxx --count 100

# # DP 1
# poetry run wandb agent lucacorbucci/private-fastshap-mnist/5zi4xexf --count 100

# # DP 2
# poetry run wandb agent lucacorbucci/private-fastshap-mnist/dykdeghn --count 100

# # DP 3
# poetry run wandb agent lucacorbucci/private-fastshap-mnist/mp4qbst7 --count 100

# DP 1
poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=19 --clipping=9.090238619181063 --eff_lambda=0.6385186563664279 --lr=0.006565895542642127 --num_samples=43 --optimizer=adam --paired_sampling=False --validation_samples=25 --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 1 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_1

# DP 2 
poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=50 --clipping=8.122879611556577 --eff_lambda=0.3196081427329157 --lr=0.0046065562994899575 --num_samples=34 --optimizer=adam --paired_sampling=True --validation_samples=35  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 2 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_2

# DP 5 
poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=47 --clipping=18.40271600052128 --eff_lambda=0.42246469836996214 --lr=0.003149427049555067 --num_samples=22 --optimizer=adam --paired_sampling=False --validation_samples=26 --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 5 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_5

# DP 10
poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=33 --clipping=2.5023544126088573 --eff_lambda=0.29915795381835464 --lr=0.01858646296184958 --num_samples=44 --optimizer=adam --paired_sampling=False --validation_samples=36 --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 10 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_10