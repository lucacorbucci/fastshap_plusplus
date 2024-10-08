# NO DP

poetry run python /home/lcorbucci/private_fastshap/train_bb/mnist/../../train_bb.py --batch_size=64 --lr=0.09729057829520492 --optimizer=sgd --epochs 10 --project_name Private_Fast_Shap --dataset_name mnist --model_name bb_mnist_NO_DP --save_model True

# DP

poetry run python /home/lcorbucci/private_fastshap/train_bb/mnist/../../train_bb.py --batch_size=417 --clipping=16.13549956271365 --lr=0.06713418430354533 --optimizer=sgd --epochs 10 --project_name Private_Fast_Shap --dataset_name mnist --epsilon 1 --model_name bb_mnist_DP_1 --save_model True