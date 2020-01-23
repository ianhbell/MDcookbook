pyflakes check_RDF.py && docker run --gpus device=1 -v "${PWD}":/output -d rumd bash -c "cd /output && python check_RDF.py"
