# # no dp
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=410 --eff_lambda=0.3898675368126264 --lr=0.011612444662325382 --num_samples=452 --optimizer=adam --paired_sampling=True --validation_samples=7  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_NO_DP.pt --dataset_name dutch --epochs 40 --save_model True --model_name explainer_dutch_NO_DP

# # 0.5
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=196 --clipping=4.105016690358632 --eff_lambda=0.9467696934202752 --lr=0.008064050674620375 --num_samples=154 --optimizer=sgd --paired_sampling=True --validation_samples=17  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_NO_DP.pt --epsilon 0.5 --dataset_name dutch --epochs 30 --save_model True --model_name explainer_dutch_DP_05

# # 1
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=486 --clipping=9.055244638515124 --eff_lambda=0.6146763128985254 --lr=0.010324168530395594 --num_samples=284 --optimizer=sgd --paired_sampling=False --validation_samples=36  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_NO_DP.pt --epsilon 1 --dataset_name dutch --epochs 30 --save_model True --model_name explainer_dutch_DP_1

# # 2
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=480 --clipping=6.957131002075527 --eff_lambda=0.08000248939180554 --lr=0.024133423419812068 --num_samples=508 --optimizer=sgd --paired_sampling=False --validation_samples=20  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_NO_DP.pt --epsilon 2 --dataset_name dutch --epochs 30 --save_model True --model_name explainer_dutch_DP_2

# # 3
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=423 --clipping=2.3938409107486427 --eff_lambda=0.9433235027377116 --lr=0.0013468060890580592 --num_samples=241 --optimizer=adam --paired_sampling=True --validation_samples=66  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_NO_DP.pt --epsilon 3 --dataset_name dutch --epochs 30 --save_model True --model_name explainer_dutch_DP_3

# 4
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=128 --clipping=17.12291037294235 --eff_lambda=0.06939694003246666 --lr=0.015603305727842888 --num_samples=390 --optimizer=sgd --paired_sampling=True --validation_samples=79  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_NO_DP.pt --epsilon 4 --dataset_name dutch --epochs 30 --save_model True --model_name explainer_dutch_DP_4
poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=355 --clipping=3.175547989357964 --eff_lambda=0.4955243482281553 --lr=0.003011761148179218 --num_samples=443 --optimizer=adam --paired_sampling=True --validation_samples=31 --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_NO_DP.pt --epsilon 4 --dataset_name dutch --epochs 30 --save_model True --model_name explainer_dutch_DP_4
# # 5
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=189 --clipping=9.46723979122331 --eff_lambda=0.8660723731700596 --lr=0.00989107748155239 --num_samples=330 --optimizer=sgd --paired_sampling=True --validation_samples=7  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_NO_DP.pt --epsilon 5 --dataset_name dutch --epochs 30 --save_model True --model_name explainer_dutch_DP_5

# # 10 
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=404 --clipping=10.898321360536531 --eff_lambda=0.6155701676453227 --lr=0.01320184309367901 --num_samples=402 --optimizer=sgd --paired_sampling=False --validation_samples=11  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_NO_DP.pt --epsilon 10 --dataset_name dutch --epochs 30 --save_model True --model_name explainer_dutch_DP_10

# # 100
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=240 --clipping=11.861467065975386 --eff_lambda=0.5534331508730826 --lr=0.014218056233729253 --num_samples=288 --optimizer=sgd --paired_sampling=True --validation_samples=96  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_NO_DP.pt --epsilon 100 --dataset_name dutch --epochs 30 --save_model True --model_name explainer_dutch_DP_100




# # no dp
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=325 --eff_lambda=0.8384754838554844 --lr=0.006234426217303377 --num_samples=272 --optimizer=adam --paired_sampling=False --validation_samples=69  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_DP_1.pt --dataset_name dutch --epochs 30 --save_model True --model_name explainer_private_BB_dutch_NO_DP

# # 0.5
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=246 --clipping=1.7150185541997611 --eff_lambda=0.04200780182888775 --lr=0.003977096960171544 --num_samples=164 --optimizer=sgd --paired_sampling=False --validation_samples=81  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_DP_1.pt --epsilon 0.5 --dataset_name dutch --epochs 30 --save_model True --model_name explainer_private_BB_dutch_DP_05

# # 1
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=461 --clipping=17.29679618860691 --eff_lambda=0.9664667411632144 --lr=0.007725962204077236 --num_samples=60 --optimizer=sgd --paired_sampling=True --validation_samples=40  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_DP_1.pt --epsilon 1 --dataset_name dutch --epochs 30 --save_model True --model_name explainer_private_BB_dutch_DP_1

# # 2
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=360 --clipping=4.284712580686345 --eff_lambda=0.040179591567129025 --lr=0.02383237698700432 --num_samples=128 --optimizer=sgd --paired_sampling=False --validation_samples=65  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_DP_1.pt --epsilon 2 --dataset_name dutch --epochs 30 --save_model True --model_name explainer_private_BB_dutch_DP_2

# # 3
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=274 --clipping=5.971421628804423 --eff_lambda=0.21907205599260848 --lr=0.021938694813758316 --num_samples=337 --optimizer=sgd --paired_sampling=True --validation_samples=69  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_DP_1.pt --epsilon 3 --dataset_name dutch --epochs 30 --save_model True --model_name explainer_private_BB_dutch_DP_3

# 4
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=479 --clipping=5.473570444410246 --eff_lambda=0.971065904520209 --lr=0.05843748651715049 --num_samples=236 --optimizer=sgd --paired_sampling=True --validation_samples=67  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_DP_1.pt --epsilon 4 --dataset_name dutch --epochs 30 --save_model True --model_name explainer_private_BB_dutch_DP_4
poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=296 --clipping=14.7398925481842 --eff_lambda=0.14925227263792473 --lr=0.027240553886186576 --num_samples=182 --optimizer=sgd --paired_sampling=True --validation_samples=67 --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_DP_1.pt --epsilon 4 --dataset_name dutch --epochs 30 --save_model True --model_name explainer_private_BB_dutch_DP_4

# # 5
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=317 --clipping=3.6346739843906866 --eff_lambda=0.10551296265474153 --lr=0.030339452576527016 --num_samples=103 --optimizer=sgd --paired_sampling=True --validation_samples=74  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_DP_1.pt --epsilon 5 --dataset_name dutch --epochs 30 --save_model True --model_name explainer_private_BB_dutch_DP_5

# # 10 
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=376 --clipping=13.7267445147089 --eff_lambda=0.319882003499779 --lr=0.018346577290221367 --num_samples=198 --optimizer=sgd --paired_sampling=False --validation_samples=47  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_DP_1.pt --epsilon 10 --dataset_name dutch --epochs 30 --save_model True --model_name explainer_private_BB_dutch_DP_10

# # 100
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=495 --clipping=3.893175760039351 --eff_lambda=0.216103928308209 --lr=0.0016118044858071591 --num_samples=171 --optimizer=adam --paired_sampling=False --validation_samples=83  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_DP_1.pt --epsilon 100 --dataset_name dutch --epochs 30 --save_model True --model_name explainer_private_BB_dutch_DP_100