

### Dockerfile (optional)

The following steps are required in order to run LOGO:

1. Install Docker
   1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support.
   2. Setup running [Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

2. Check that AlphaFold will be able to use a GPU by running:

```
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

The output of this command should show a list of your GPUs. If it doesn't, check if you followed all steps correctly when setting up the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) or take a look at the following [NVIDIA Docker issue](https://github.com/NVIDIA/nvidia-docker/issues/1447#issuecomment-801479573).

3. Bulid LOGO Docker image envirment

```
git clone https://github.com/melobio/LOGO.git
cd LOGO-master/docker
docker build -f Dockerfile -t mgi_logo .
```
The configuration process is as follows: 
![image](https://user-images.githubusercontent.com/27897166/136744049-74ea5335-d9a5-47d6-8fab-96c44d488a23.png)



4. Launch LOGO docker images

Set up this repository directory to share to docker images

```
nvidia-docker run -it -v [absolutepath]/LOGO-master:/home/LOGO-master mgi_logo:latest bash
```
For example:
![image](https://user-images.githubusercontent.com/27897166/136750812-0a28c154-430c-4e8c-a9f3-89b65628044b.png)


5. Try to run demo script

```
source activate logo
bash 01_Pre-training_Model/xxx.sh
```
For example:
![image](https://user-images.githubusercontent.com/82862879/136742008-40407ed4-40fc-436e-b000-61894330fcd6.png)


