## Docker 
A Dockerfile is included to allow you to run the entire project in a containerized environment. This environment already has the game installed and headless rendering enabled.
To build the container, run the following command in the project directory:
```bash
docker build -t <name> .
```
All necessary dependencies (like Python) are pre-installed within the container. For detailed information on GPU rendering options and the container's configuration, refer to the base image documentation [here](https://github.com/SgSiegens/TMNF-Docker).
## Using Containers on the bwUniCluster at KIT 

### Connecting to the cluster
We assume you already have an account to access the bwUniCluster. If not, please follow the instructions provided [here](https://bwidm.scc.kit.edu/). To log in to the cluster, you must first be connected 
to the KIT network—either by using a KIT campus network or by connecting through the KIT VPN. Once connected, you can access the cluster via SSH:
```bash
ssh ka_<username>@uc3.scc.kit.edu
```
You will then be prompted to enter your **one-time password (OTP)**, which you set up during the account registration process. After entering the OTP, you will be asked to provide your regular account password.
For a more detailed login tutorial, please refer to the documentation **[here](https://wiki.bwhpc.de/e/BwUniCluster3.0/Login)**.

### Transfering Data to the cluster
To work on the cluster, you will eventually need to transfer local files such as container images or datasets. Before doing so, decide where the data should be stored on the cluster:
- Workspace
  Suitable for temporary data or data that is backed up elsewhere (e.g., trained models uploaded via Weights & Biases). Workspaces expire after a fixed period and are automatically deleted.

- Home directory 
  Recommended for important data that must be preserved permanently. You can create dedicated subdirectories here to store container images, datasets, or long-term training results.

More information about the cluster filesystems and workspace system is available in this [documentation](https://www.nhr.kit.edu/userdocs/horeka/filesystems/).
To copy your files to your desired location on the cluster, you can use the **`scp`** command (see documentation **[here](https://wiki.bwhpc.de/e/Data_Transfer/SCP)**).
For example, you can run:
```bash
scp myfile <username>@<remotehost>:/some/remote/directory
```

For KIT the remote host is usually `uc3.scc.kit.edu`. **Make sure to run this command from your local machine**, not from within the cluster**.

### Running Jobs on the Cluster
The cluster uses [Slurm](https://slurm.schedmd.com/overview.html) for job scheduling and resource management. The BWUniCluster provides an extensive documentation on how to run jobs on the cluster which you can find 
[here](https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs#Queues_on_bwUniCluster_3.0). Additionally, it is helpful to check which partitions are available for job execution see their [hardware page](https://wiki.bwhpc.de/e/BwUniCluster3.0/Hardware_and_Architecture).
**Some Advice**
Use only the partitions listed on the hardware page when launching or debugging jobs. In particular, when using the `dev_` prefix for development/testing partitions, make sure the corresponding base partition actually exists on that page. 
We found that using partitions not listed there sometimes redirected us to what appear to be older parts of the cluster, where resources (such as GPUs) did not work correctly. But it could also just been a mistake from our side.

### Enroot and Pyxis 
In this section, we describe how to use NVIDIA Enroot and [Pyxis](https://github.com/NVIDIA/pyxis) to run containers on the bwUniCluster. This is only a minimal guide to help you get started. For a more detailed tutorial, refer 
to the [Enroot section](https://wiki.bwhpc.de/e/BwUniCluster3.0/Containers#ENROOT) of the bwUniCluster documentation or [here](https://www.nhr.kit.edu/userdocs/horeka/containers/).

#### Converting a Local Image with Enroot
On your local machine, after building the Docker image, you need to convert it into an Enroot-compatible image. This is done using the Enroot **import** command (assuming Enroot is already installed). In this example, we import 
a local Docker image, but Enroot also supports pulling images directly from Docker Hub. To convert your local Docker image into an Enroot image run:
```bash
enroot import -o <enroot_img_store_path>.sqsh dockerd://<your_image>:<tag>
```
For more details on how to use enroot, see the official documentation [here](https://github.com/NVIDIA/enroot/blob/main/doc/usage.md).

#### Running the Container with Pyxis
Pyxis is a Slurm plugin that enables running privileged containerized tasks directly through Slurm. In practice, this means you can submit normal Slurm jobs, but with additional parameters 
that ensure your command is executed inside a specified container. For more detailed information, see the [usage section](https://github.com/NVIDIA/pyxis?tab=readme-ov-file#usage).

When running on the bwUniCluster, one important flag must always be set: `--container-mounts`. The required value for the BwUniCluster3 is:

```bash
--container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,/usr/lib64/slurm:/usr/lib64/slurm,/usr/lib64/libhwloc.so:/usr/lib64/libhwloc.so,/usr/lib64/libhwloc.so.15:/usr/lib64/libhwloc.so.15
```
If you store your container in a workspace, Pyxis will not find it by default. To make it work, you **must** set:

```bash
export XDG_DATA_HOME=$(ws_find <workspace>)
export ENROOT_DATA_PATH=$(ws_find <workspace>)/enroot
```

#### Example Slurm Script
Below is an example Slurm job script that:

- uses a development node for testing `dev_gpu_h100`
- runs inside a workspace named `first_tmnf_test`
- uses a local Enroot container image (`tmnf_wandb.sqsh`)
- mounts required paths
- exposes Weights & Biases API key inside the container
  (you can obtain your key at [https://wandb.ai/authorize](https://wandb.ai/authorize))

```bash
#!/bin/bash
#SBATCH --container-image=./tmnf_wandb.sqsh
#SBATCH --container-mounts= /etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,/usr/lib64/slurm:/usr/lib64/slurm,/usr/lib64/libhwloc.so:/usr/lib64/libhwloc.so,/usr/lib64/libhwloc.so.15:/usr/lib64/libhwloc.so.15
#SBATCH -- no-container-entrypoint
#SBATCH --container-workdir=/home/wineuser/trackmania_gym
#SBATCH --time=00:07:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

# Expose W&B key inside the container
export WANDB_API_KEY=<your_api_key>

python scripts/sb3_train.py
```

### Singularity