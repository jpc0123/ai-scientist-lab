# CPU-only HOW plugin author image. Not the RGB-T GPU experiment image.
# Does not request NVIDIA devices. Build explicitly; the worker never auto-pulls.
#
#   docker build -t scientist-lab/dsh-plugin-author:dev docker/dsh-plugin-author
#   docker build -t scientist-lab/dsh-plugin-author:dev --build-arg INSTALL_SDK=1 docker/dsh-plugin-author
#
ARG PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.12-slim-bookworm
FROM ${PYTHON_IMAGE}

WORKDIR /workspace

ARG INSTALL_SDK=0
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
RUN if [ "$INSTALL_SDK" = "1" ]; then \
      pip install --no-cache-dir -i "$PIP_INDEX" deepseek-harness-sdk ; \
    fi

COPY entrypoint.py cordis.yml.tmpl /opt/dsh-plugin-author/
ENTRYPOINT ["python", "/opt/dsh-plugin-author/entrypoint.py"]
