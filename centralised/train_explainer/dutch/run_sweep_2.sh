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
  poetry run wandb agent $SWEEP_ID --project "$PROJECT_NAME" --count 40
}



run_sweep_and_agent "dutch_NO_DP_private_model"
run_sweep_and_agent "dutch_DP_05_private_model"
run_sweep_and_agent "dutch_DP_1_private_model"
run_sweep_and_agent "dutch_DP_2_private_model"
run_sweep_and_agent "dutch_DP_3_private_model"
run_sweep_and_agent "dutch_DP_4_private_model"
run_sweep_and_agent "dutch_DP_5_private_model"
run_sweep_and_agent "dutch_DP_10_private_model"
run_sweep_and_agent "dutch_DP_100_private_model"







