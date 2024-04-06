# Go Tensor development environment

This repository provides a Go development container environment for working TensorFlow, and OpenCV.
The aim is to have the devcontainer build all components for ARM-based processors like Apple Silicon
or Raspberry Pi. GPU support is not available.

## Prerequisites

Before getting started, make sure you have the following installed on your machine:

- Docker: [Install Docker](https://docs.docker.com/get-docker/)
- Visual Studio Code: [Install Visual Studio Code](https://code.visualstudio.com/)

## Getting Started

To use this dev container environment, follow these steps:

1. Clone this repository to your local machine.
2. Open the cloned repository in Visual Studio Code.
3. If prompted, click on the "Reopen in Container" button in the bottom-right corner of Visual Studio Code.
4. Wait for the dev container to build. This may take a few minutes.
5. Once the dev container is ready, you can start writing your Golang code using TensorFlow and OpenCV.

## Usage

Inside the dev container, you can use any Golang package, including TensorFlow and OpenCV, to build your machine learning and computer vision applications. You can also install additional dependencies as needed.

Here are some example steps to get started:

1. Create a new Go file in the `src` directory of your cloned repository.
2. Import the necessary packages, such as `github.com/tensorflow/tensorflow` for TensorFlow and `gocv.io/x/gocv` for OpenCV.
3. Write your Go code to perform machine learning or computer vision tasks using TensorFlow and OpenCV.
4. Build and run your Go program using the `go` command. For example, you can run `go run src/your_file.go` to execute your code.

### Webcam streaming

Due to the differences in how MacOS/Windows and Linux systems handle devices, it's difficult to mount a video device to a Linux container from the host OS. To capture webcam video and process it with OpenCV, it first needs to be FFMPEG streamed.

1. Install FFMPEG using Homebrew:

```shell
brew install ffmpeg
```

2. Start the stream using FFMPEG:

```shell
ffmpeg \         
  -f avfoundation \
  -pix_fmt yuyv422 \
  -video_size 640x480 \
  -framerate 30 \
  -i "0:none" \
  -vf format=yuyv422 \
  -vcodec libx264 -maxrate 2000k \
  -bufsize 2000k \
  -tune zerolatency \
  -f mpegts udp://127.0.0.1:9988
```

3. Run demo app in the container and check proccessed output at <http://localhost:8080>
