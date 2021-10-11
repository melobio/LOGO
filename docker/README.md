

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
cd LOGO-master
docker build -f docker/Dockerfile -t mgi_logo .
```
The configuration process is as follows: 
![image](https://user-images.githubusercontent.com/82862879/136741879-7bf9eeb6-c053-4812-82b0-36e9d217af9b.png)

![image](https://user-images.githubusercontent.com/82862879/136741749-60907687-9a29-47c7-a6fb-c0d0f821e0a7.png)

![image](https://user-images.githubusercontent.com/27897166/136743444-b53cddd8-a1ee-41e9-8518-8e5d06e543b8.png)


4. Launch LOGO docker images

Set up this repository directory to share to docker images

```
nvidia-docker run -it mgi_logo:laster -v [absolutepath]/LOGO-master:/home/LOGO-master bash
```
For example:
![image](https://user-images.githubusercontent.com/27897166/136743817-e1c99b93-0d15-4e11-a468-fd1a4bed49f4.png)


5. Try to run demo script

```
source activate logo
bash 01_Pre-training_Model/xxx.sh
```
For example:
![image](https://user-images.githubusercontent.com/82862879/136742008-40407ed4-40fc-436e-b000-61894330fcd6.png)


