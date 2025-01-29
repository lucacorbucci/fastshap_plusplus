PROJECT_NAME="private-fastshap" # swap out globally

run_sweep_and_agent () {
  # Set the SWEEP_NAME variable
  SWEEP_NAME="$1"

  
  # Run the wandb sweep command and store the output in a temporary file
  poetry run wandb sweep --project "$PROJECT_NAME" --name "$SWEEP_NAME" "$SWEEP_NAME.yaml" >temp_output.txt 2>&1
  
  # Extract the sweep ID using awk
  SWEEP_ID=$(awk '/wandb agent/{ match($0, /wandb agent (.+)/, arr); print arr[1]; }' temp_output.txt)

  # Remove the temporary output file
#   rm temp_output.txt
  
  # Run the wandb agent command
  poetry run wandb agent $SWEEP_ID --project "$PROJECT_NAME" --count 30
}




run_sweep_and_agent "income_NO_DP"

/usr/bin/env poetry run python ../../train_explainer_comparison.py --batch_size=5000 --eff_lambda=0.3682434556803168 --lr=0.06565670240049083 --num_samples=512 --optimizer=adam --paired_sampling=True --validation_samples=48 --sweep True --project_name private-fastshap --surrogate ./income_surrogate_NO_DP.pth --dataset_name income --epochs 30