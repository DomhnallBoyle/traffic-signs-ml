FROM ubuntu:18.04 as base

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libsm6 \
    libxext6 \
    libxrender1

COPY . /opt
WORKDIR /opt
RUN pip3 install -r requirements.txt

FROM base as development
RUN pip3 install watchdog==0.9.0
# ENTRYPOINT watchmedo auto-restart --recursive --pattern="*.py" --directory="." ./start.sh  # doesn't work
ENTRYPOINT ./start.sh

FROM base as production
ENTRYPOINT ./start.sh