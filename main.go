package main

import (
	"flag"
	"log"
	"net/http"
	"time"

	"github.com/hybridgroup/mjpeg"
	tf "github.com/wamuir/graft/tensorflow"
	"github.com/wamuir/graft/tensorflow/op"
	"gocv.io/x/gocv"

	_ "github.com/u2takey/ffmpeg-go"
)

func main() {
	printVersion()

	var streamURL string

	flag.StringVar(&streamURL, "url", "udp://@:9988", "vedeo capture stream URL")
	flag.Parse()

	if streamURL == "" {
		flag.Usage()

		return
	}

	rtp, err := gocv.OpenVideoCaptureWithAPI(streamURL, gocv.VideoCaptureFFmpeg)
	if err != nil {
		panic(err)
	}

	defer rtp.Close()

	log.Printf("Capturing from %s\n", streamURL)

	stream := mjpeg.NewStream()

	go func() {
		img := gocv.NewMat()
		defer img.Close()

		for {
			if ok := rtp.Read(&img); !ok {
				return
			}

			if img.Empty() {
				continue
			}

			buf, _ := gocv.IMEncode(".jpg", img)
			stream.UpdateJPEG(buf.GetBytes())
			buf.Close()
		}
	}()

	http.Handle("/", stream)

	server := &http.Server{
		Addr:         ":8080",
		ReadTimeout:  60 * time.Second,
		WriteTimeout: 60 * time.Second,
	}

	log.Println("Processed at http://localhost:8080/")

	if err := server.ListenAndServe(); err != nil {
		panic(err)
	}
}

func printVersion() {
	scope := op.NewScope()
	ver := op.Const(scope, tf.Version())

	graph, err := scope.Finalize()
	if err != nil {
		panic(err)
	}

	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		panic(err)
	}

	output, err := sess.Run(nil, []tf.Output{ver}, nil)
	if err != nil {
		panic(err)
	}

	log.Printf("TensorFlow version %s\n", output[0].Value())
	log.Printf("OpenCV version %s\n", gocv.Version())
}
