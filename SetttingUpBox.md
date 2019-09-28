Install the driver for nvidia directly from Ubuntu 18.04 (https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-18-04-bionic-beaver-linux)
```
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
```

Confirm that you get something useful from nvidia-smi on the host
```
ian@XXX:~/Code/XXX$ nvidia-smi
Wed Sep 25 10:14:25 2019
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 435.21       Driver Version: 435.21       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce RTX 2080    Off  | 00000000:01:00.0 Off |                  N/A |
|  0%   32C    P8     1W / 245W |     34MiB /  7982MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce RTX 2080    Off  | 00000000:02:00.0 Off |                  N/A |
|  0%   27C    P8     1W / 245W |      1MiB /  7982MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0       975      G   /usr/lib/xorg/Xorg                            33MiB |
+-----------------------------------------------------------------------------+

```

Add the runtime to docker (https://github.com/nvidia/nvidia-container-runtime#systemd-drop-in-file)
``` 
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/override.conf <<EOF
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd --host=fd:// --add-runtime=nvidia=/usr/bin/nvidia-container-runtime
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```

Run the nvidia container, make sure you can talk to card
```
ian@XXX:~/Code/XXX$ docker run --gpus all nvidia/cuda:10.1-base nvidia-smi
Wed Sep 25 16:15:28 2019
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 435.21       Driver Version: 435.21       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce RTX 2080    Off  | 00000000:01:00.0 Off |                  N/A |
|  0%   31C    P8     1W / 245W |     34MiB /  7982MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce RTX 2080    Off  | 00000000:02:00.0 Off |                  N/A |
|  0%   27C    P8     1W / 245W |      1MiB /  7982MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```


On the host, make GPU cards exclusive per process
```
nvidia-smi -c 1
```