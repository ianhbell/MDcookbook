
``
cd docker
``

Build:
```
docker build -t rumd .
```

Run, exposing the GPU of index 1 to the container, and mapping current folder to /output inside the container
```
docker run --gpus device=1 -v "${PWD}":/output -it -t rumd bash
```

Run, exposing all the GPU to the container:
```
docker run --gpus all -v "${PWD}":/output -it -t rumd bash
```

Debugging
```
docker run -it --gpus all -t rumd bash
```