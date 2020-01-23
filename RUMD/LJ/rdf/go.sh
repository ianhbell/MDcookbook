pyflakes check_RDF.py && cd .. && docker run --gpus device=1 -v "${PWD}":/output -d rumd bash -c "cd /output/rdf && python check_RDF.py"
