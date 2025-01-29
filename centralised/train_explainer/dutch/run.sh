# # no dp
poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=167 --eff_lambda=0.2864863741835173 --lr=0.005290221394037948 --num_samples=297 --optimizer=adam --paired_sampling=True --validation_samples=100  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_NO_DP.pt --dataset_name dutch --epochs 40 # --save_model True --model_name dutch_explainer_NO_DP

# # # 0.5
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=418 --clipping=4.276528827447256 --eff_lambda=0.7208838059741337 --lr=0.008541045256903718 --num_samples=387 --optimizer=sgd --paired_sampling=False --validation_samples=20  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_NO_DP.pt --epsilon 0.5 --dataset_name dutch --epochs 30 --save_model True --model_name dutch_explainer_DP_05

# # # 1
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=306 --clipping=16.608112232295845 --eff_lambda=0.5811685797882901 --lr=0.00497439535214859 --num_samples=388 --optimizer=sgd --paired_sampling=True --validation_samples=87  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_NO_DP.pt --epsilon 1 --dataset_name dutch --epochs 30 --save_model True --model_name dutch_explainer_DP_1

# # # 2
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=374 --clipping=2.0122700120352572 --eff_lambda=0.2014171502037997 --lr=0.009632360367864194 --num_samples=382 --optimizer=adam --paired_sampling=False --validation_samples=19  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_NO_DP.pt --epsilon 2 --dataset_name dutch --epochs 30 --save_model True --model_name dutch_explainer_DP_2

# # # 3
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=223 --clipping=13.831276088726383 --eff_lambda=0.01639423701765763 --lr=0.009110092041838274 --num_samples=480 --optimizer=sgd --paired_sampling=True --validation_samples=45  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_NO_DP.pt --epsilon 3 --dataset_name dutch --epochs 30 --save_model True --model_name dutch_explainer_DP_3

# # 4
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=265 --clipping=13.1903917763671 --eff_lambda=0.157532501877076 --lr=0.011051087647069718 --num_samples=99 --optimizer=sgd --paired_sampling=True --validation_samples=30  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_NO_DP.pt --epsilon 4 --dataset_name dutch --epochs 30 --save_model True --model_name dutch_explainer_DP_4

# # # 5
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=379 --clipping=4.601982173350951 --eff_lambda=0.6498730071856551 --lr=0.004049715235039287 --num_samples=268 --optimizer=adam --paired_sampling=False --validation_samples=56  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_NO_DP.pt --epsilon 5 --dataset_name dutch --epochs 30 --save_model True --model_name dutch_explainer_DP_5

# # # 10 
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=329 --clipping=6.698840907925163 --eff_lambda=0.16899511864999672 --lr=0.007106663687790336 --num_samples=168 --optimizer=sgd --paired_sampling=False --validation_samples=49  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_NO_DP.pt --epsilon 10 --dataset_name dutch --epochs 30 --save_model True --model_name dutch_explainer_DP_10

# # # 100
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=266 --clipping=6.170919650433163 --eff_lambda=0.01709652331509559 --lr=0.003370793656558964 --num_samples=322 --optimizer=sgd --paired_sampling=True --validation_samples=54  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_NO_DP.pt --epsilon 100 --dataset_name dutch --epochs 30 --save_model True --model_name dutch_explainer_DP_100




# # # no dp
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=227 --eff_lambda=0.41272685282962074 --lr=0.016513174366542194 --num_samples=342 --optimizer=adam --paired_sampling=True --validation_samples=14  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_DP_1.pt --dataset_name dutch --epochs 30 --save_model True --model_name dutch_explainer_NO_DP_private_model

# # # 0.5
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=200 --clipping=4.799818264166736 --eff_lambda=0.4269451037716038 --lr=0.0031695560891964797 --num_samples=474 --optimizer=sgd --paired_sampling=False --validation_samples=13  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_DP_1.pt --epsilon 0.5 --dataset_name dutch --epochs 30 --save_model True --model_name dutch_explainer_private_model_DP_05

# # # 1
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=422 --clipping=11.806603684363903 --eff_lambda=0.3559386284279431 --lr=0.010589154150474574 --num_samples=349 --optimizer=sgd --paired_sampling=False --validation_samples=37  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_DP_1.pt --epsilon 1 --dataset_name dutch --epochs 30 --save_model True --model_name dutch_explainer_private_model_DP_1

# # # 2
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=176 --clipping=17.922726132278264 --eff_lambda=0.6099600503084284 --lr=0.005406962619107489 --num_samples=278 --optimizer=sgd --paired_sampling=False --validation_samples=13  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_DP_1.pt --epsilon 2 --dataset_name dutch --epochs 30 --save_model True --model_name dutch_explainer_private_model_DP_2

# # # 3
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=151 --clipping=11.9655167140166 --eff_lambda=0.3429234400007125 --lr=0.005464242665969368 --num_samples=275 --optimizer=sgd --paired_sampling=False --validation_samples=92  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_DP_1.pt --epsilon 3 --dataset_name dutch --epochs 30 --save_model True --model_name dutch_explainer_private_model_DP_3

# # 4
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=371 --clipping=1.550454043595109 --eff_lambda=0.7756663367371964 --lr=0.002127062479834799 --num_samples=269 --optimizer=adam --paired_sampling=False --validation_samples=6  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_DP_1.pt --epsilon 4 --dataset_name dutch --epochs 30 --save_model True --model_name dutch_explainer_private_model_DP_4

# # # 5
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=280 --clipping=5.960148586671648 --eff_lambda=0.08953856179768882 --lr=0.003459696372521044 --num_samples=511 --optimizer=adam --paired_sampling=True --validation_samples=81  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_DP_1.pt --epsilon 5 --dataset_name dutch --epochs 30 --save_model True --model_name dutch_explainer_private_model_DP_5

# # # 10 
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=278 --clipping=13.530791065996484 --eff_lambda=0.9750053385705372 --lr=0.00042695670697017466 --num_samples=508 --optimizer=adam --paired_sampling=False --validation_samples=21  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_DP_1.pt --epsilon 10 --dataset_name dutch --epochs 30 --save_model True --model_name dutch_explainer_private_model_DP_10

# # # 100
# poetry run python /home/lcorbucci/private_fastshap/train_explainer/dutch/../../train_explainer.py --batch_size=325 --clipping=4.769931107621256 --eff_lambda=0.8917616039313365 --lr=0.001200126017731423 --num_samples=46 --optimizer=adam --paired_sampling=False --validation_samples=96  --project_name private-fastshap --surrogate ../../artifacts/dutch/surrogate/dutch_surrogate_DP_1.pt --epsilon 100 --dataset_name dutch --epochs 30 --save_model True --model_name dutch_explainer_private_model_DP_100