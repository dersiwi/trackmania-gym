FROM condaforge/mambaforge:latest AS conda-builder
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev build-essential libevdev-dev && \
    rm -rf /var/lib/apt/lists/*

COPY environment.yml /tmp/environment.yml
RUN mamba env create -p /env -f /tmp/environment.yml && \
    conda clean -afy

FROM ghcr.io/sgsiegens/tmnf-docker-vulkan:latest
USER root

COPY --chown=${USER}:${USER} --from=conda-builder /env /env

WORKDIR /home/${USER}/trackmania_gym

COPY --chown=${USER}:${USER} . .

# we excluded the outputs dir through our .dockerignore to avoid copy tons of pretained models, so we need to include it here again manually
RUN mkdir -p checkpoints logs outputs runs wandb \
    && chown -R ${USER}:${USER} checkpoints logs outputs runs wandb

RUN cat <<EOF > configs/platforms.yaml
platforms:
  os: linux
  home: /home/${USER}
  tmloader: /home/${USER}/.wine/drive_c/Program_Files_x86/TmNationsForever/TMLoader.exe
  plugin: /home/${USER}/.wine/drive_c/users/${USER}/Documents/TMInterface/
  map_dir: /home/${USER}/.wine/drive_c/users/${USER}/Documents/TmForever/Tracks/Challenges
  device: cuda
EOF

# we need this in order to use python without activating envs
ENV PATH="/env/bin:$PATH"

USER ${USER}
WORKDIR /home/${USER}/trackmania_gym
CMD ["bash"]
