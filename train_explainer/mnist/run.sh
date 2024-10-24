# NO DP
poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=236 --eff_lambda=0.5354794915050585 --lr=0.023339074206445504 --num_samples=23 --optimizer=adam --paired_sampling=True --validation_samples=8 --sweep True --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_NO_DP

# # EPS 1
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=214 --clipping=17.60775715113185 --eff_lambda=0.5251462801574658 --lr=0.0014980515217183055 --num_samples=9 --optimizer=adam --paired_sampling=False --validation_samples=5  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 1 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_1

# # Eps 2
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=224 --clipping=13.746970632116568 --eff_lambda=0.8531145822948039 --lr=0.012781748674444942 --num_samples=23 --optimizer=adam --paired_sampling=False --validation_samples=4  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 2 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_2

# # EPS 3 -> rifare
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=158 --clipping=7.5645780093236255 --eff_lambda=0.2544563185100134 --lr=0.02574110104088493 --num_samples=22 --optimizer=adam --paired_sampling=True --validation_samples=23  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 3 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_3
# # EPS 4
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=96 --clipping=17.52658446752294 --eff_lambda=0.11216486140818294 --lr=0.01687356021979186 --num_samples=12 --optimizer=adam --paired_sampling=False --validation_samples=4  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 4 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_4

# # EPS 5
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=125 --clipping=2.351803035006659 --eff_lambda=0.5204389921829404 --lr=0.0044973225164079925 --num_samples=10 --optimizer=adam --paired_sampling=True --validation_samples=19  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 5 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_5

# # EPS 10
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=171 --clipping=6.730771957227484 --eff_lambda=0.0003916533322777527 --lr=0.003034979086056616 --num_samples=23 --optimizer=adam --paired_sampling=True --validation_samples=21  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 10 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_10

# # EPS 100
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=240 --clipping=7.8684708035801485 --eff_lambda=0.002949517995771145 --lr=0.01977038856989097 --num_samples=2 --optimizer=adam --paired_sampling=False --validation_samples=23  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 100 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_100

# # NO DP - Private Model
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=83 --eff_lambda=0.6564790784687966 --lr=0.0745649519086709 --num_samples=13 --optimizer=adam --paired_sampling=False --validation_samples=22  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_DP_1.pt --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_NO_DP_private_model

# # EPS 1 - Private Model
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=215 --clipping=10.94146644056173 --eff_lambda=0.5418724434929674 --lr=0.008383501293876177 --num_samples=16 --optimizer=adam --paired_sampling=False --validation_samples=20  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_DP_1.pt --epsilon 1 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_private_model_DP_1

# # Eps 2 - Private Model
# poetry run python python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=105 --clipping=7.823134403023345 --eff_lambda=0.12124036362521906 --lr=0.019594968645607024 --num_samples=9 --optimizer=adam --paired_sampling=True --validation_samples=7  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_DP_1.pt --epsilon 2 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_private_model_DP_2

# # EPS 3 - Private Model
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=92 --clipping=8.112164074888094 --eff_lambda=0.025413421477373133 --lr=0.000610922447111603 --num_samples=7 --optimizer=adam --paired_sampling=False --validation_samples=5  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_DP_1.pt --epsilon 3 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_private_model_DP_3

# # EPS 4 - Private Model
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=176 --clipping=13.27317382680211 --eff_lambda=0.3771140043297197 --lr=0.002818625708490318 --num_samples=15 --optimizer=adam --paired_sampling=False --validation_samples=25  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_DP_1.pt --epsilon 4 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_private_model_DP_4

# # EPS 5 - Private Model
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=253 --clipping=9.542429371877374 --eff_lambda=0.7883575279364585 --lr=0.010001036542725656 --num_samples=8 --optimizer=adam --paired_sampling=False --validation_samples=25  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_DP_1.pt --epsilon 5 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_private_model_DP_5

# # EPS 10 - Private Model
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=234 --clipping=11.58924201476636 --eff_lambda=0.8395803735607735 --lr=0.008693809075837987 --num_samples=20 --optimizer=adam --paired_sampling=True --validation_samples=17  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_DP_1.pt --epsilon 10 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_private_model_DP_10
# # EPS 100 - Private Model
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=64 --clipping=1.0724812738554332 --eff_lambda=0.6979216677047301 --lr=0.035219143795856896 --num_samples=25 --optimizer=adam --paired_sampling=True --validation_samples=12  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_DP_1.pt --epsilon 100 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_private_model_DP_100

# # EPS 3 -> rifare
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=158 --clipping=7.5645780093236255 --eff_lambda=0.2544563185100134 --lr=0.02574110104088493 --num_samples=22 --optimizer=adam --paired_sampling=True --validation_samples=23  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 3 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_private_model_DP_3
