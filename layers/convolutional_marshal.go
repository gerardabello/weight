package layers

import (
	"archive/tar"
	"encoding/json"
	"errors"
	"io"
	"io/ioutil"

	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/marshaling"
	"gitlab.com/gerardabello/weight/tensor"
)

func (l *ConvolutionalLayer) Marshal(writer io.Writer) error {
	tarfile := tar.NewWriter(writer)
	defer tarfile.Close()

	//save info
	err := writeInfoTar(
		tarfile,
		&map[string]interface{}{
			"id":          l.id,
			"inputWidth":  l.inputWidth,
			"inputHeight": l.inputHeight,
			"inputDepth":  l.inputDepth,
			"nKernels":    l.weights.Size[3],
			"kernelPadX":  (l.weights.Size[0] - 1) / 2,
			"kernelPadY":  (l.weights.Size[1] - 1) / 2,
			"strideX":     l.strideX,
			"strideY":     l.strideY,
			"padX":        l.padX,
			"padY":        l.padY,
		},
	)
	if err != nil {
		return err
	}

	err = writeTensorTar(tarfile, l.bias, "bias")
	if err != nil {
		return err
	}

	err = writeTensorTar(tarfile, l.weights, "weights")
	if err != nil {
		return err
	}

	return nil

}

const ConvolutionalName = "convolutional"

func init() {
	marshaling.RegisterFormat(ConvolutionalName, UnmarshalConvolutionalLayer)
}

func (l *ConvolutionalLayer) GetName() string {
	return ConvolutionalName
}

func UnmarshalConvolutionalLayer(reader io.Reader) (weight.MarshalLayer, error) {
	tr := tar.NewReader(reader)
	// Iterate through the files in the archive.

	var id string
	var inputWidth, inputHeight, inputDepth, nKernels, kernelPadX, kernelPadY, strideX, strideY, padX, padY int

	var weights *tensor.Tensor
	var bias *tensor.Tensor

	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			// end of tar archive
			break
		}
		if err != nil {
			return nil, err
		}

		switch hdr.Name {
		case "id":
			buf, err := ioutil.ReadAll(tr)
			if err != nil {
				return nil, errors.New("id file read: " + err.Error())
			}
			id = string(buf)

		case "info":
			info := map[string]interface{}{}
			dec := json.NewDecoder(tr)
			err = dec.Decode(&info)
			if err == io.EOF {
				continue
			}
			if err != nil {
				return nil, err
			}

			var ok bool

			id, ok = info["id"].(string)
			if !ok {
				return nil, errors.New("Unmarshal convolutional: error parsing info file")
			}

			inputWidth, ok = InterfaceFloatToInt(info["inputWidth"])
			if !ok {
				return nil, errors.New("Unmarshal convolutional: error parsing info file")
			}
			inputHeight, ok = InterfaceFloatToInt(info["inputHeight"])
			if !ok {
				return nil, errors.New("Unmarshal convolutional: error parsing info file")
			}
			inputDepth, ok = InterfaceFloatToInt(info["inputDepth"])
			if !ok {
				return nil, errors.New("Unmarshal convolutional: error parsing info file")
			}
			nKernels, ok = InterfaceFloatToInt(info["nKernels"])
			if !ok {
				return nil, errors.New("Unmarshal convolutional: error parsing info file")
			}
			kernelPadX, ok = InterfaceFloatToInt(info["kernelPadX"])
			if !ok {
				return nil, errors.New("Unmarshal convolutional: error parsing info file")
			}
			kernelPadY, ok = InterfaceFloatToInt(info["kernelPadY"])
			if !ok {
				return nil, errors.New("Unmarshal convolutional: error parsing info file")
			}
			strideX, ok = InterfaceFloatToInt(info["strideX"])
			if !ok {
				return nil, errors.New("Unmarshal convolutional: error parsing info file")
			}
			strideY, ok = InterfaceFloatToInt(info["strideY"])
			if !ok {
				return nil, errors.New("Unmarshal convolutional: error parsing info file")
			}
			padX, ok = InterfaceFloatToInt(info["padX"])
			if !ok {
				return nil, errors.New("Unmarshal convolutional: error parsing info file")
			}
			padY, ok = InterfaceFloatToInt(info["padY"])
			if !ok {
				return nil, errors.New("Unmarshal convolutional: error parsing info file")
			}

		case "weights":
			weights, err = tensor.Unmarshal(tr)
			if err != nil {
				return nil, err
			}
		case "bias":
			bias, err = tensor.Unmarshal(tr)
			if err != nil {
				return nil, err

			}
		default:
			return nil, errors.New("Unrecognized file " + hdr.Name)
		}

	}

	l := NewConvolutionalLayer(inputWidth, inputHeight, inputDepth, nKernels, kernelPadX, kernelPadY, strideX, strideY, padX, padY)

	l.weights = weights
	l.bias = bias
	l.id = id

	return l, nil
}
