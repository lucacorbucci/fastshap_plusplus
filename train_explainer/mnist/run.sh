# NO DP
poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=97 --eff_lambda=0.07562284273890618 --lr=0.06750531066569665 --num_samples=12 --optimizer=adam --paired_sampling=False --validation_samples=16  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_NO_DP

# EPS 1
poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=116 --clipping=12.530192710936952 --eff_lambda=0.6503903941853094 --lr=0.004593793881071256 --num_samples=19 --optimizer=adam --paired_sampling=False --validation_samples=4  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 1 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_1

# Eps 2
poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=171 --clipping=12.752480627760638 --eff_lambda=0.31674506398743885 --lr=0.02271903054067072 --num_samples=20 --optimizer=adam --paired_sampling=True --validation_samples=3  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 2 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_2

# EPS 3
poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=162 --clipping=6.210843357362176 --eff_lambda=0.5394146921677679 --lr=0.007092056717582722 --num_samples=17 --optimizer=adam --paired_sampling=False --validation_samples=15  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 3 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_3

# EPS 4
poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=134 --clipping=15.781627078306114 --eff_lambda=0.9059141099802278 --lr=0.01436287032524976 --num_samples=13 --optimizer=adam --paired_sampling=True --validation_samples=10  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 4 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_4

# EPS 5
poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=254 --clipping=4.378869172758143 --eff_lambda=0.6230649347158923 --lr=0.008594911696175983 --num_samples=5 --optimizer=adam --paired_sampling=False --validation_samples=14  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 5 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_5

# EPS 10
poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=149 --clipping=9.01243829189571 --eff_lambda=0.4867441253826494 --lr=0.004362100426631279 --num_samples=3 --optimizer=adam --paired_sampling=False --validation_samples=21  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 10 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_10

# EPS 100
poetry run python /home/lcorbucci/private_fastshap/train_explainer/mnist/../../train_explainer.py --batch_size=143 --clipping=8.454306224785352 --eff_lambda=0.5638054359557221 --lr=0.004105992868071844 --num_samples=18 --optimizer=adam --paired_sampling=False --validation_samples=17  --project_name private-fastshap --surrogate ../../artifacts/mnist/surrogate/mnist_surrogate_NO_DP.pt --epsilon 100 --dataset_name mnist --epochs 20 --save_model True --model_name mnist_explainer_DP_100