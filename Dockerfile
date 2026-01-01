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

RUN mkdir -p /home/${USER}/trackmania_gym \
    && chown ${USER}:${USER} /home/${USER}/trackmania_gym 

RUN mkdir -p /home/${USER}/configs \
    && cat <<EOF > /home/${USER}/configs/platforms.yaml
platforms:
  os: linux
  home: /home/${USER}
  tmloader: ${WINEPREFIX:-/home/${USER}/.wine}/drive_c/Program_Files_x86/TmNationsForever/TMLoader.exe
  plugin: ${WINEPREFIX:-/home/${USER}/.wine}/drive_c/users/${USER}/Documents/TMInterface/
  map_dir: ${WINEPREFIX:-/home/${USER}/.wine}/drive_c/users/${USER}/Documents/TmForever/Tracks/Challenges
  device: cuda
EOF

RUN chmod 755 /home/${USER}/configs \
    && chmod 644 /home/${USER}/configs/platforms.yaml

# we need this in order to use python without activating envs
ENV PATH="/env/bin:$PATH"

USER ${USER}
WORKDIR /home/${USER}/trackmania_gym
CMD ["bash"]
