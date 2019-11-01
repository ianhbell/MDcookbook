docker run --gpus device=1 -v "${PWD}":/output -d rumd bash -c "cd /output && python validate_Meier.py"
