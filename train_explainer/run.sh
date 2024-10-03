# no dp
poetry run python /home/lcorbucci/private_fast_shap/train_explainer/../explainer.py --batch_size=266 --eff_lambda=0.21812205214214184 --lr=0.00309442906157747 --normalization=additive --num_samples=229 --optimizer=adam --paired_sampling=True --validation_samples=79 --project_name private-fastshap --surrogate ../train_surrogate/surrogate_Dutch_NO_DP.pt --dataset_name dutch --epochs 40 --save_model True --model_name explainer_dutch_NO_DP

# 0.5
poetry run python /home/lcorbucci/private_fast_shap/train_explainer/../explainer.py --batch_size=208 --clipping=1.445869485638363 --eff_lambda=0.33361080824288003 --lr=0.0049457687638892085 --normalization=none --num_samples=425 --optimizer=sgd --paired_sampling=False --validation_samples=58 --project_name private-fastshap --surrogate ../train_surrogate/surrogate_Dutch_NO_DP.pt --epsilon 0.5 --dataset_name dutch --epochs 40  --save_model True --model_name explainer_dutch_DP_05

# 1
poetry run python /home/lcorbucci/private_fast_shap/train_explainer/../explainer.py --batch_size=457 --clipping=1.6835066923265989 --eff_lambda=0.4328492108517296 --lr=0.04637524475741835 --normalization=additive --num_samples=267 --optimizer=sgd --paired_sampling=True --validation_samples=49 --project_name private-fastshap --surrogate ../train_surrogate/surrogate_Dutch_NO_DP.pt --epsilon 1 --dataset_name dutch --epochs 40 --save_model True --model_name explainer_dutch_DP_1

# 2
poetry run python /home/lcorbucci/private_fast_shap/train_explainer/../explainer.py --batch_size=496 --clipping=10.818620885329446 --eff_lambda=0.15233844722606416 --lr=0.09387018759145258 --normalization=multiplicative --num_samples=65 --optimizer=sgd --paired_sampling=True --validation_samples=41 --project_name private-fastshap --surrogate ../train_surrogate/surrogate_Dutch_NO_DP.pt --epsilon 2 --dataset_name dutch --epochs 40 --save_model True --model_name explainer_dutch_DP_2

# 3
poetry run python /home/lcorbucci/private_fast_shap/train_explainer/../explainer.py --batch_size=329 --clipping=2.0592855267892425 --eff_lambda=0.3279820087647628 --lr=0.031222867793025577 --normalization=additive --num_samples=311 --optimizer=sgd --paired_sampling=False --validation_samples=79 --project_name private-fastshap --surrogate ../train_surrogate/surrogate_Dutch_NO_DP.pt --epsilon 3 --dataset_name dutch --epochs 40 --save_model True --model_name explainer_dutch_DP_3

# 4
poetry run python /home/lcorbucci/private_fast_shap/train_explainer/../explainer.py --batch_size=435 --clipping=13.195465923530142 --eff_lambda=0.7361607700094133 --lr=0.01944085307466264 --normalization=additive --num_samples=282 --optimizer=sgd --paired_sampling=False --validation_samples=21 --project_name private-fastshap --surrogate ../train_surrogate/surrogate_Dutch_NO_DP.pt --epsilon 4 --dataset_name dutch --epochs 40 --save_model True --model_name explainer_dutch_DP_4

# 5
poetry run python /home/lcorbucci/private_fast_shap/train_explainer/../explainer.py --batch_size=422 --clipping=15.36280812740901 --eff_lambda=0.6584033736799015 --lr=0.07885184459937508 --normalization=multiplicative --num_samples=436 --optimizer=sgd --paired_sampling=True --validation_samples=44 --project_name private-fastshap --surrogate ../train_surrogate/surrogate_Dutch_NO_DP.pt --epsilon 5 --dataset_name dutch --epochs 40 --save_model True --model_name explainer_dutch_DP_5

# 10 
poetry run python /home/lcorbucci/private_fast_shap/train_explainer/../explainer.py --batch_size=209 --clipping=6.701146376790746 --eff_lambda=0.4015013503524309 --lr=0.050445857289251335 --normalization=additive --num_samples=196 --optimizer=sgd --paired_sampling=False --validation_samples=38 --project_name private-fastshap --surrogate ../train_surrogate/surrogate_Dutch_NO_DP.pt --epsilon 10 --dataset_name dutch --epochs 40 --save_model True --model_name explainer_dutch_DP_10

# 100
poetry run python /home/lcorbucci/private_fast_shap/train_explainer/../explainer.py --batch_size=355 --clipping=19.718370891635995 --eff_lambda=0.49559391231463024 --lr=0.03478455526964593 --normalization=additive --num_samples=82 --optimizer=sgd --paired_sampling=True --validation_samples=20 --project_name private-fastshap --surrogate ../train_surrogate/surrogate_Dutch_NO_DP.pt --epsilon 100 --dataset_name dutch --epochs 40 --save_model True --model_name explainer_dutch_DP_100