Build:
```
docker build -t rumd .
```
Run:
```
docker run --gpus all rumd 
```
Debugging
```
docker run -it --gpus all -t rumd bash
```