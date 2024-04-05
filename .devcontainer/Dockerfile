# syntax=docker/dockerfile:1

FROM mcr.microsoft.com/vscode/devcontainers/go:latest

RUN sudo sed -i 's/^Components: main$/& contrib non-free/' /etc/apt/sources.list.d/debian.sources
RUN sudo apt-get update

ARG GoCvVersion=0.36.1

RUN git clone -b v${GoCvVersion} https://github.com/hybridgroup/gocv.git && cd gocv && \
    make install && cd ../.. && rm -rf gocv*

ARG TensorFlowVersion=2.13.0 

RUN wget https://github.com/ika-rwth-aachen/libtensorflow_cc/releases/download/v${TensorFlowVersion}/libtensorflow-cc_${TensorFlowVersion}_$(dpkg --print-architecture).deb && \
    dpkg -i libtensorflow-cc_${TensorFlowVersion}_$(dpkg --print-architecture).deb && \
    cp /usr/local/lib/libtensorflow_cc.so /usr/local/lib/libtensorflow.so && \
    ldconfig
