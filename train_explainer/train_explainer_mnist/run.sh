# NO DP
poetry run python /home/lcorbucci/private_fast_shap/train_explainer_adult/../explainer.py --batch_size=268 --eff_lambda=0.633832857339435 --lr=0.002827688107573299 --normalization=none --num_samples=315 --optimizer=adam --paired_sampling=True --validation_samples=79  --project_name private-fastshap --surrogate ../train_surrogate_adult/adult_surrogate_NO_DP.pt --dataset_name adult --epochs 40 --save_model True --model_name explainer_adult_NO_DP

# EPS 1
poetry run python /home/lcorbucci/private_fast_shap/train_explainer_adult/../explainer.py --batch_size=385 --clipping=3.134766234151794 --eff_lambda=0.28739079680374224 --lr=0.0406013877843143 --normalization=additive --num_samples=431 --optimizer=sgd --paired_sampling=True --validation_samples=67  --project_name private-fastshap --surrogate ../train_surrogate_adult/adult_surrogate_NO_DP.pt --epsilon 1 --dataset_name adult --epochs 40 --save_model True --model_name explainer_adult_DP_1

# Eps 2
poetry run python /home/lcorbucci/private_fast_shap/train_explainer_adult/../explainer.py --batch_size=380 --clipping=1.3283242440079652 --eff_lambda=0.10090902309093296 --lr=0.08549791336691955 --normalization=additive --num_samples=80 --optimizer=sgd --paired_sampling=True --validation_samples=18  --project_name private-fastshap --surrogate ../train_surrogate_adult/adult_surrogate_NO_DP.pt --epsilon 2 --dataset_name adult --epochs 40 --save_model True --model_name explainer_adult_DP_2

# EPS 3
poetry run python /home/lcorbucci/private_fast_shap/train_explainer_adult/../explainer.py --batch_size=346 --clipping=5.793745152406884 --eff_lambda=0.715713465530328 --lr=0.04793589138290233 --normalization=multiplicative --num_samples=190 --optimizer=sgd --paired_sampling=False --validation_samples=56  --project_name private-fastshap --surrogate ../train_surrogate_adult/adult_surrogate_NO_DP.pt --epsilon 3 --dataset_name adult --epochs 40 --save_model True --model_name explainer_adult_DP_3

# EPS 4
poetry run python /home/lcorbucci/private_fast_shap/train_explainer_adult/../explainer.py --batch_size=131 --clipping=1.3532774326215895 --eff_lambda=0.3712005535682623 --lr=0.03507153858331423 --normalization=additive --num_samples=92 --optimizer=sgd --paired_sampling=False --validation_samples=28  --project_name private-fastshap --surrogate ../train_surrogate_adult/adult_surrogate_NO_DP.pt --epsilon 4 --dataset_name adult --epochs 40 --save_model True --model_name explainer_adult_DP_4

# EPS 5
poetry run python /home/lcorbucci/private_fast_shap/train_explainer_adult/../explainer.py --batch_size=481 --clipping=10.506989451992832 --eff_lambda=0.26183944116441527 --lr=0.06884529906681457 --normalization=multiplicative --num_samples=127 --optimizer=sgd --paired_sampling=False --validation_samples=20  --project_name private-fastshap --surrogate ../train_surrogate_adult/adult_surrogate_NO_DP.pt --epsilon 5 --dataset_name adult --epochs 40 --save_model True --model_name explainer_adult_DP_5

# EPS 10
poetry run python /home/lcorbucci/private_fast_shap/train_explainer_adult/../explainer.py --batch_size=387 --clipping=9.201791246805527 --eff_lambda=0.8518478362913453 --lr=0.09996822939851108 --normalization=multiplicative --num_samples=348 --optimizer=sgd --paired_sampling=True --validation_samples=16  --project_name private-fastshap --surrogate ../train_surrogate_adult/adult_surrogate_NO_DP.pt --epsilon 10 --dataset_name adult --epochs 40 --save_model True --model_name explainer_adult_DP_10

# EPS 100
poetry run python /home/lcorbucci/private_fast_shap/train_explainer_adult/../explainer.py --batch_size=456 --clipping=12.013283575127542 --eff_lambda=0.2631020604993558 --lr=0.0642979494675246 --normalization=additive --num_samples=424 --optimizer=sgd --paired_sampling=True --validation_samples=70  --project_name private-fastshap --surrogate ../train_surrogate_adult/adult_surrogate_NO_DP.pt --epsilon 100 --dataset_name adult --epochs 40 --save_model True --model_name explainer_adult_DP_100