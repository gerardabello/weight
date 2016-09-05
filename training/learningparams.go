package training

//ParamUpdateMethod defines the method used to update parameters in each layer
type ParamUpdateMethod int

const (
	//Momentum uses the momentum configuration parameter as a low-pass filter to smooth the gradient descent
	Momentum ParamUpdateMethod = iota
	//AdaDelta updates each parameter independently, without a global learning rate
	AdaDelta
	//Adam is a mix of AdaDelta + Momentum
	Adam
)

//LearningConfig contains the parameters used in the learning process
type LearningConfig struct {
	Method            ParamUpdateMethod
	LearningRateStart float64
	LearningRateEnd   float64

	Epochs      int
	BatchSize   int
	WeightDecay float64
	Momentum    float64
}
