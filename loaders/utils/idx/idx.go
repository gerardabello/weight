package idx

import (
	"encoding/binary"
	"io"
)

const (
	Uint8DataType   int8 = 0x08
	Int8DataType    int8 = 0x09
	Int16DataType   int8 = 0x0b
	Int32DataType   int8 = 0x0c
	Float32DataType int8 = 0x0d
	Float64DataType int8 = 0x0e
)

type Header struct {
	Zeros         int16
	DataType      int8
	NumDimensions int8
}

type Reader struct {
	// Idx header for read-convenience
	Header *Header

	Dimensions []int

	// Underlying io.Reader
	reader io.Reader
}

func NewReader(r io.Reader) (rr *Reader, err error) {
	header := new(Header)
	err = binary.Read(r, binary.BigEndian, header)
	if err != nil {
		return
	}

	dims32 := make([]int32, header.NumDimensions)
	err = binary.Read(r, binary.BigEndian, dims32)
	if err != nil {
		return
	}

	dims := make([]int, header.NumDimensions)

	for i, d := range dims32 {
		dims[i] = int(d)
	}

	rr = &Reader{
		Header:     header,
		Dimensions: dims,
		reader:     r,
	}
	return
}

func (rr *Reader) Read(p []byte) (int, error) {
	return rr.reader.Read(p)
}

func (rr *Reader) ReadUint8(p []uint8) error {
	return binary.Read(rr.reader, binary.BigEndian, p)
}

func (rr *Reader) ReadInt8(p []int8) error {
	return binary.Read(rr.reader, binary.BigEndian, p)
}

func (rr *Reader) ReadInt16(p []int16) error {
	return binary.Read(rr.reader, binary.BigEndian, p)
}

func (rr *Reader) ReadInt32(p []int32) error {
	return binary.Read(rr.reader, binary.BigEndian, p)
}

func (rr *Reader) ReadFloat32(p []float32) error {
	return binary.Read(rr.reader, binary.BigEndian, p)
}

func (rr *Reader) ReadFloat64(p []float64) error {
	return binary.Read(rr.reader, binary.BigEndian, p)
}

type Writer struct {
	// Idx header for write-convenience
	header *Header

	// Underlying io.Writer
	writer io.Writer
}

func NewWriter(w io.Writer, dataType int8, dimensions []int32) (ww *Writer) {
	header := &Header{
		DataType:      dataType,
		NumDimensions: int8(len(dimensions)),
	}
	binary.Write(w, binary.BigEndian, header)
	binary.Write(w, binary.BigEndian, dimensions)
	ww = &Writer{
		header: header,
		writer: w,
	}
	return
}

func (ww *Writer) Write(p []byte) (int, error) {
	return ww.writer.Write(p)
}

func (ww *Writer) WriteUint8(p []uint8) error {
	return binary.Write(ww.writer, binary.BigEndian, p)
}

func (ww *Writer) WriteInt8(p []int8) error {
	return binary.Write(ww.writer, binary.BigEndian, p)
}

func (ww *Writer) WriteInt16(p []int16) error {
	return binary.Write(ww.writer, binary.BigEndian, p)
}

func (ww *Writer) WriteInt32(p []int32) error {
	return binary.Write(ww.writer, binary.BigEndian, p)
}

func (ww *Writer) WriteFloat32(p []float32) error {
	return binary.Write(ww.writer, binary.BigEndian, p)
}

func (ww *Writer) WriteFloat64(p []float64) error {
	return binary.Write(ww.writer, binary.BigEndian, p)
}
