package main

import (
	"log"

	tf "github.com/wamuir/graft/tensorflow"
	"github.com/wamuir/graft/tensorflow/op"
	"gocv.io/x/gocv"

	_ "github.com/u2takey/ffmpeg-go"
)

func main() {
	s := op.NewScope()
	ver := op.Const(s, tf.Version())
	graph, err := s.Finalize()
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
