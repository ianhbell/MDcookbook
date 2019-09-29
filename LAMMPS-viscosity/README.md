Build:
```
docker build -t lampvis .
```
Run:
```
docker run --gpus all lampvis 
```
Debugging
```
docker run -it --gpus all -t lampvis bash
```