package layers

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"

	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/tensor"
)

type ConvolutionalLayer struct {
	BaseLayer

	inputWidth, inputHeight, inputDepth int
	padX, padY                          int
	strideX, strideY                    int
	strideJumpsX, strideJumpsY          int

	im2colTmp *tensor.Tensor
}

//NewConvolutionalLayer does what it says
//
// inputWidth: input width
// inputHeight: input height
// inputDepth: input depth
// nKernels: number of kernels
// kernelPadX: kernel width pad
// kernelPadY: kernel height pad
// strideX: stride in x
// strideY: stride in y
// padX: padding in x
// padY: padding in y
//
// Instead of passing the kernel size, you specify the size as a padding of a kernel of one pixel. This is to avoid the possibility of a kernel size that has no center pixel. It's also easier to create a layer that does not change the input size if kernelPad = pad
//
// The filter cannot be bigger than the image so: kernelPadX*2+1 <= inputWidth && kernelPadY*2+1 <= inputHeight
// The stride must divide the image in equal integer parts:  ((inputWidth + padX*2) - (1+kernelPadX*2)) % (strideX+1) == 0 && ((inputHeight + padY*2) - (1+kernelPadY*2)) % (strideY+1) == 0
// If you want the output area to be the same as the input: kernelPadX == padX && kernelPadY == padY
// More padding than spatial extend makes no sense so: padX <= kernelPadX && padY < kernelPadY
func NewConvolutionalLayer(inputWidth, inputHeight, inputDepth, nKernels, kernelPadX, kernelPadY, strideX, strideY, padX, padY int) *ConvolutionalLayer {
	if inputHeight <= 0 || inputWidth <= 0 || inputDepth <= 0 {
		panic("Input sizes must be bigger than 0")
	}

	if nKernels <= 0 {
		panic("Number of kernels must be bigger than 0")
	}

	if kernelPadX < 0 || kernelPadY < 0 {
		panic("Kernel pads must be positive")
	}

	if strideX <= 0 || strideY <= 0 {
		panic("Strides must be bigger than 0")
	}

	kernelWidth := kernelPadX*2 + 1
	kernelHeight := kernelPadY*2 + 1

	if !(kernelWidth <= inputWidth && kernelHeight <= inputHeight) {
		panic("Kernel cannot be bigger than image")
	}

	if padX > kernelPadX || padY > kernelPadY {
		panic("A padding bigger than the kernel padding makes no sense")
	}

	if ((inputWidth+(padX-kernelPadX)*2)+strideX-1)%strideX != 0 {
		panic(fmt.Sprintf("StrideX of %d does not fit", strideX))
	}
	if ((inputHeight+(padY-kernelPadY)*2)+strideY-1)%strideY != 0 {
		panic(fmt.Sprintf("StrideY of %d does not fit", strideY))
	}

	strideJumpsX := ((inputWidth + (padX-kernelPadX)*2) + strideX - 1) / strideX
	strideJumpsY := ((inputHeight + (padY-kernelPadY)*2) + strideY - 1) / strideY

	l := &ConvolutionalLayer{}

	l.inputWidth = inputWidth
	l.inputHeight = inputHeight
	l.inputDepth = inputDepth
	l.padX = padX
	l.padY = padY

	l.strideX = strideX
	l.strideY = strideY

	l.strideJumpsX = strideJumpsX
	l.strideJumpsY = strideJumpsY

	//initialize bias and kernel
	l.bias = tensor.NewTensor(nKernels)
	l.biasGrad = tensor.NewTensor(l.bias.Size...)

	l.weights = tensor.NewTensor(kernelWidth, kernelHeight, inputDepth, nKernels)
	l.weightsGrad = tensor.NewTensor(l.weights.Size...)

	stdev := math.Sqrt(2.0 / float64(kernelHeight*kernelWidth*inputDepth))
	for i := range l.weights.Values {
		//Initialize weights with uniform random from -variance to variance
		l.weights.Values[i] = rand.NormFloat64() * stdev
	}

	outputSize := []int{l.strideJumpsX, l.strideJumpsY, nKernels}
	inputSize := []int{l.inputWidth, l.inputHeight, l.inputDepth}

	l.BaseLayer.Init(inputSize, outputSize)
	l.id = "Conv-" + l.id

	return l
}

func NewSquareConvolutionalLayer(inputSize, inputDepth, nKernels, kernelPad, stride, padding int) *ConvolutionalLayer {
	return NewConvolutionalLayer(inputSize, inputSize, inputDepth, nKernels, kernelPad, kernelPad, stride, stride, padding, padding)
}

//CreateSlave creates a slave of the ConvolutionalLayer. See EnslaverLayer in package weight for more information on layer slaves.
func (l *ConvolutionalLayer) CreateSlave() weight.Layer {
	kernelPadX := (l.weights.Size[0] - 1) / 2
	kernelPadY := (l.weights.Size[1] - 1) / 2

	nl := NewConvolutionalLayer(l.inputWidth, l.inputHeight, l.inputDepth, l.GetOutputSize()[2], kernelPadX, kernelPadY, l.strideX, l.strideY, l.padX, l.padY)

	nl.id = l.ID()

	nl.weights = l.weights
	nl.bias = l.bias

	return nl
}

//Activate takes and input tensor and computes an output tensor where each value is the sum of the convolutions of the different input depths using different kernels.
func (l *ConvolutionalLayer) Activate(input *tensor.Tensor) (*tensor.Tensor, error) {
	l.mutex.Lock()
	err := l.BaseLayer.Activate(input)
	if err != nil {
		return nil, err
	}

	l.ConvIm2Col(input, l.weights, &l.output, l.weights.Size[0], l.weights.Size[1], l.padX, l.padY, l.strideX, l.strideY)

	l.mutex.Unlock()
	return &l.output, nil
}

func (l *ConvolutionalLayer) BackPropagate(err *tensor.Tensor) (*tensor.Tensor, error) {
	l.mutex.Lock()

	//We cannot back propagate a layer that has not been activated first
	if l.lastInput == nil {
		return nil, fmt.Errorf("weight.Layer cannot propagate error because it has not been activated or it has not been configured to retain inputs")
	}

	if len(err.Size) != len(l.GetOutputSize()) {
		return nil, errors.New("Error gradient has wrong number of dimensions")
	}
	for i := 0; i < len(err.Size); i++ {
		if err.Size[i] != l.GetOutputSize()[i] {
			return nil, errors.New("Error gradient has wrong size")
		}
	}

	e := l.BaseLayer.BackPropagate(err)
	if e != nil {
		return nil, e
	}

	lastInputs := l.lastInput.Values

	kernelPadX := (l.weights.Size[0] - 1) / 2
	kernelPadY := (l.weights.Size[1] - 1) / 2

	startX := kernelPadX - l.padX
	startY := kernelPadY - l.padY

	if startX < 0 || startY < 0 {
		return nil, fmt.Errorf("Padding is bigger than kernel padding")
	}

	//Store sizes
	ws1 := l.weights.Size[1]
	ws0 := l.weights.Size[0]
	ws2 := l.weights.Size[2]

	is1 := l.propagation.Size[1]
	is0 := l.propagation.Size[0]

	//////
	for d := 0; d < l.weights.Size[3]; d++ {
		var f = l.weights.Slice(d)
		var fg = l.weightsGrad.Slice(d)
		x := -kernelPadX
		y := -kernelPadY
		for ay := 0; ay < err.Size[1]; ay++ {
			y += l.strideY
			x = -kernelPadX
			for ax := 0; ax < err.Size[0]; ax++ {
				x += l.strideX
				var grad = err.GetVal(ax, ay, d)

				for fd := 0; fd < ws2; fd++ {
					for fy := 0; fy < ws1; fy++ {
						oy := y + fy
						if oy >= 0 && oy < is1 {
							for fx := 0; fx < ws0; fx++ {
								ox := x + fx
								if ox >= 0 && ox < is0 {
									//TODO optimize this
									wp := fd*(ws0*ws1) + fy*(ws0) + fx
									pp := fd*(is0*is1) + oy*(is0) + ox

									l.propagation.Values[pp] += grad * f.Values[wp]
									fg.Values[wp] += grad * lastInputs[pp]

								}
							}
						}
					}
				}
				l.biasGrad.Values[d] += grad
			}
		}
	}

	l.mutex.Unlock()
	return &l.propagation, nil
}

func (l *ConvolutionalLayer) ConvIm2Col(data, ker, out *tensor.Tensor, kernelSizeX, kernelSizeY, padX, padY, strideX, strideY int) {

	if l.im2colTmp == nil {
		//If its the first time, initialize the temp tensor to store the volume/image as columns
		l.im2colTmp = tensor.NewTensor(im2colSize(data.Size[0], data.Size[1], data.Size[2], kernelSizeX, kernelSizeY, padX, padY, strideX, strideY))
	}

	//Transform input volume/image as a matrix
	im2col(data.Values, l.im2colTmp.Values, data.Size[0], data.Size[1], data.Size[2], kernelSizeX, kernelSizeY, padX, padY, strideX, strideY)

	//Add bias to output
	n := out.GetNumberOfValues()
	dStride := out.Size[0] * out.Size[1]
	for i := 0; i < n; i++ {
		out.Values[i] = l.bias.Values[i/dStride]
	}

	//Create blas matrix with underlying tensor values
	matImg := blas64.General{
		Rows:   l.im2colTmp.Size[1],
		Cols:   l.im2colTmp.Size[0],
		Stride: l.im2colTmp.Size[0],
		Data:   l.im2colTmp.Values,
	}

	matKer := blas64.General{
		Rows:   ker.Size[3],
		Cols:   ker.Size[0] * ker.Size[1] * ker.Size[2],
		Stride: ker.Size[0] * ker.Size[1] * ker.Size[2],
		Data:   ker.Values,
	}

	matOut := blas64.General{
		Rows:   ker.Size[3],
		Cols:   l.im2colTmp.Size[0],
		Stride: l.im2colTmp.Size[0],
		Data:   out.Values,
	}

	//Calculate matrix multiply. We will add to matOut that already has the bias
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, matKer, matImg, 1, matOut)
}

func im2colSize(width int, height int, channels int, kernel_w int, kernel_h int, pad_w int, pad_h int, stride_w int, stride_h int) (int, int) {
	var height_col int = (height+2*pad_h-kernel_h)/stride_h + 1
	var width_col int = (width+2*pad_w-kernel_w)/stride_w + 1
	var channels_col int = channels * kernel_h * kernel_w

	return width_col * height_col, channels_col
}

func im2col(im []float64, col []float64, width int, height int, channels int, kernel_w int, kernel_h int, pad_w int, pad_h int, stride_w int, stride_h int) {
	var height_col int = (height+2*pad_h-kernel_h)/stride_h + 1
	var width_col int = (width+2*pad_w-kernel_w)/stride_w + 1
	var channels_col int = channels * kernel_h * kernel_w

	for c := 0; c < channels_col; c++ {
		var w_offset int = c % kernel_w
		var h_offset int = (c / kernel_w) % kernel_h
		var c_im int = c / (kernel_h * kernel_w)
		for h := 0; h < height_col; h++ {
			for w := 0; w < width_col; w++ {
				var h_pad int = h*stride_h - pad_h + h_offset
				var w_pad int = w*stride_w - pad_w + w_offset
				if h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width {
					//t.Values[(c*height_col+h)*width_col+w] = img[(c_im*height+h_pad)*width+w_pad]
					col[(c*height_col+h)*width_col+w] = im[(c_im*height+h_pad)*width+w_pad]
				} else {
					col[(c*height_col+h)*width_col+w] = 0
				}
			}
		}
	}
}

func col2im(col []float64, channels int, height int, width int, patch_h int, patch_w int, pad_h int, pad_w int, stride_h int, stride_w int, im []float64) {
	var height_col int = (height+2*pad_h-patch_h)/stride_h + 1
	var width_col int = (width+2*pad_w-patch_w)/stride_w + 1
	var channels_col int = channels * patch_h * patch_w
	for c := 0; c < channels_col; c++ {
		var w_offset int = c % patch_w
		var h_offset int = (c / patch_w) % patch_h
		var c_im int = c / patch_h / patch_w
		for h := 0; h < height_col; h++ {
			for w := 0; w < width_col; w++ {
				var h_pad int = h*stride_h - pad_h + h_offset
				var w_pad int = w*stride_w - pad_w + w_offset
				if h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width {
					im[(c_im*height+h_pad)*width+w_pad] += col[(c*height_col+h)*width_col+w]
				}
			}
		}
	}
}
