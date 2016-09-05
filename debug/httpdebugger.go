package debug

import (
	"log"
	"net/http"

	"github.com/gorilla/websocket"
)

const host string = "localhost:8888"

type HttpDebugger struct {
	layerInfo <-chan []*LayerInfo
	trainInfo <-chan *TrainInfo
	testInfo  <-chan *TestInfo
}

func (panel *HttpDebugger) Debug(status <-chan string, layerInfo <-chan []*LayerInfo, trainInfo <-chan *TrainInfo, testInfo <-chan *TestInfo) {
	println("Starting server at " + host + "...")
	panel.layerInfo = layerInfo
	panel.trainInfo = trainInfo
	panel.testInfo = testInfo

	fs := http.FileServer(http.Dir("static"))

	http.HandleFunc("/train", panel.train)
	http.HandleFunc("/test", panel.test)
	http.HandleFunc("/layers", panel.layers)
	http.Handle("/", fs)
	log.Fatal(http.ListenAndServe(host, nil))
}

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
} // use default options

func (panel *HttpDebugger) train(w http.ResponseWriter, r *http.Request) {
	c, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Print("upgrade:", err)
		return
	}
	defer c.Close()

	for trainStats := range panel.trainInfo {
		err := c.WriteJSON(trainStats)
		if err != nil {
			log.Println("write:", err)
		}
	}
}

func (panel *HttpDebugger) test(w http.ResponseWriter, r *http.Request) {
	c, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Print("upgrade:", err)
		return
	}
	defer c.Close()

	for testStats := range panel.testInfo {
		err := c.WriteJSON(testStats)
		if err != nil {
			log.Println("write:", err)
		}
	}
}

func (panel *HttpDebugger) layers(w http.ResponseWriter, r *http.Request) {
	c, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Print("upgrade:", err)
		return
	}
	defer c.Close()

	for layerStats := range panel.layerInfo {
		err := c.WriteJSON(layerStats)
		if err != nil {
			log.Println("write:", err)
		}
	}
}
