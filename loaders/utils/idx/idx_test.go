package idx

import (
	"bytes"
	"testing"
)

func TestNewReader(t *testing.T) {
	sample := bytes.NewBuffer([]byte{
		0x00, 0x00, 0x08, 0x03,
		0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x00, 0x00, 0x02,
		0x01, 0x02, 0x03, 0x04,
	})
	rd, err := NewReader(sample)
	if err != nil {
		t.Fatal(err)
	}
	if rd.Dimensions[0] != 1 || rd.Dimensions[1] != 2 || rd.Dimensions[2] != 2 {
		t.Fatal("Got unexpect dimensions")
	}
}

func TestNewReader_Error1(t *testing.T) {
	_, err := NewReader(new(bytes.Buffer))
	if err == nil {
		t.Fatal("Expected error from nil io.Reader")
	}
}

func TestNewReader_Error2(t *testing.T) {
	sample := bytes.NewBuffer([]byte{
		0x00, 0x00, 0x08, 0x03,
	})
	_, err := NewReader(sample)
	if err == nil {
		t.Fatal("Expected error from buffer: too short")
	}
}

func TestRead(t *testing.T) {
	sample := bytes.NewBuffer([]byte{
		0x00, 0x00, 0x08, 0x03,
		0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x00, 0x00, 0x02,
		0x01, 0x02, 0x03, 0x04,
	})
	rd, err := NewReader(sample)
	if err != nil {
		t.Fatal(err)
	}
	el := make([]byte, 4)
	_, err = rd.Read(el)
	if err != nil {
		t.Fatal(err)
	}
	if el[0] != 1 || el[1] != 2 || el[2] != 3 || el[3] != 4 {
		t.Fatal("Got unexpected results", el)
	}
}

func TestReadUint8(t *testing.T) {
	sample := bytes.NewBuffer([]byte{
		0x00, 0x00, 0x08, 0x03,
		0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x00, 0x00, 0x02,
		0x01, 0x02, 0x03, 0x04,
	})
	rd, err := NewReader(sample)
	if err != nil {
		t.Fatal(err)
	}
	el := make([]uint8, 4)
	err = rd.ReadUint8(el)
	if err != nil {
		t.Fatal(err)
	}
	if el[0] != 1 || el[1] != 2 || el[2] != 3 || el[3] != 4 {
		t.Fatal("Got unexpected results", el)
	}
}

func TestReadInt8(t *testing.T) {
	sample := bytes.NewBuffer([]byte{
		0x00, 0x00, 0x09, 0x03,
		0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x00, 0x00, 0x02,
		0x01, 0x02, 0x03, 0x04,
	})
	rd, err := NewReader(sample)
	if err != nil {
		t.Fatal(err)
	}
	el := make([]int8, 4)
	err = rd.ReadInt8(el)
	if err != nil {
		t.Fatal(err)
	}
	if el[0] != 1 || el[1] != 2 || el[2] != 3 || el[3] != 4 {
		t.Fatal("Got unexpected results", el)
	}
}

func TestReadInt16(t *testing.T) {
	sample := bytes.NewBuffer([]byte{
		0x00, 0x00, 0x0B, 0x03,
		0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x01, 0x00, 0x02,
		0x00, 0x03, 0x00, 0x04,
	})
	rd, err := NewReader(sample)
	if err != nil {
		t.Fatal(err)
	}
	el := make([]int16, 4)
	err = rd.ReadInt16(el)
	if err != nil {
		t.Fatal(err)
	}
	if el[0] != 1 || el[1] != 2 || el[2] != 3 || el[3] != 4 {
		t.Fatal("Got unexpected results", el)
	}
}

func TestReadInt32(t *testing.T) {
	sample := bytes.NewBuffer([]byte{
		0x00, 0x00, 0x0C, 0x03,
		0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x00, 0x00, 0x03,
		0x00, 0x00, 0x00, 0x04,
	})
	rd, err := NewReader(sample)
	if err != nil {
		t.Fatal(err)
	}
	el := make([]int32, 4)
	err = rd.ReadInt32(el)
	if err != nil {
		t.Fatal(err)
	}
	if el[0] != 1 || el[1] != 2 || el[2] != 3 || el[3] != 4 {
		t.Fatal("Got unexpected results", el)
	}
}

func TestReadFloat32(t *testing.T) {
	sample := bytes.NewBuffer([]byte{
		0x00, 0x00, 0x0D, 0x03,
		0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x00, 0x00, 0x02,
		0x3f, 0x80, 0x00, 0x00,
		0x40, 0x00, 0x00, 0x00,
		0x40, 0x40, 0x00, 0x00,
		0x40, 0x80, 0x00, 0x00,
	})
	rd, err := NewReader(sample)
	if err != nil {
		t.Fatal(err)
	}
	el := make([]float32, 4)
	err = rd.ReadFloat32(el)
	if err != nil {
		t.Fatal(err)
	}
	if el[0] != 1 || el[1] != 2 || el[2] != 3 || el[3] != 4 {
		t.Fatal("Got unexpected results", el)
	}
}

func TestReadFloat64(t *testing.T) {
	sample := bytes.NewBuffer([]byte{
		0x00, 0x00, 0x0E, 0x03,
		0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x00, 0x00, 0x02,
		0x3f, 0xf0, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x40, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x40, 0x08, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x40, 0x10, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
	})
	rd, err := NewReader(sample)
	if err != nil {
		t.Fatal(err)
	}
	el := make([]float64, 4)
	err = rd.ReadFloat64(el)
	if err != nil {
		t.Fatal(err)
	}
	if el[0] != 1 || el[1] != 2 || el[2] != 3 || el[3] != 4 {
		t.Fatal("Got unexpected results", el)
	}
}

func TestNewWriter(t *testing.T) {
	buf := new(bytes.Buffer)
	wr := NewWriter(buf, Uint8DataType, []int32{1, 2, 2})
	if wr.header.DataType != Uint8DataType {
		t.Fatal("Got unexpect data type")
	}
	if wr.header.NumDimensions != 3 {
		t.Fatal("Got unexpect dimensions")
	}
}

func TestWrite(t *testing.T) {
	buf := new(bytes.Buffer)
	wr := NewWriter(buf, Uint8DataType, []int32{1, 2, 2})

	sample := []byte{0x01, 0x02, 0x03, 0x04}
	n, err := wr.Write(sample)
	if err != nil {
		t.Fatal(err)
	}
	if n != 4 {
		t.Fatal("Wrote incorrect number of bytes", n)
	}

	b := buf.Bytes()

	header := b[0:16]
	expectedHeader := []byte{
		0x00, 0x00, 0x08, 0x03,
		0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x00, 0x00, 0x02,
	}
	if !bytes.Equal(header, expectedHeader) {
		t.Fatal("Wrote unexpected header", header, expectedHeader)
	}

	el := b[16:]
	if !bytes.Equal(el, sample) {
		t.Fatal("Wrote unexpected elements", el)
	}
}

func TestWriteUint8(t *testing.T) {
	buf := new(bytes.Buffer)
	wr := NewWriter(buf, Uint8DataType, []int32{1, 2, 2})

	sample := []uint8{1, 2, 3, 4}
	err := wr.WriteUint8(sample)
	if err != nil {
		t.Fatal(err)
	}

	b := buf.Bytes()

	header := b[0:16]
	expectedHeader := []byte{
		0x00, 0x00, 0x08, 0x03,
		0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x00, 0x00, 0x02,
	}
	if !bytes.Equal(header, expectedHeader) {
		t.Fatal("Wrote unexpected header", header, expectedHeader)
	}

	el := b[16:]
	if !bytes.Equal(el, sample) {
		t.Fatal("Wrote unexpected elements", el)
	}
}

func TestWriteInt8(t *testing.T) {
	buf := new(bytes.Buffer)
	wr := NewWriter(buf, Int8DataType, []int32{1, 2, 2})

	sample := []int8{1, 2, 3, 4}
	err := wr.WriteInt8(sample)
	if err != nil {
		t.Fatal(err)
	}

	b := buf.Bytes()

	header := b[0:16]
	expectedHeader := []byte{
		0x00, 0x00, 0x09, 0x03,
		0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x00, 0x00, 0x02,
	}
	if !bytes.Equal(header, expectedHeader) {
		t.Fatal("Wrote unexpected header", header, expectedHeader)
	}

	el := b[16:]
	if !bytes.Equal(el, []byte{0x01, 0x02, 0x03, 0x04}) {
		t.Fatal("Wrote unexpected elements", el)
	}
}

func TestWriteInt16(t *testing.T) {
	buf := new(bytes.Buffer)
	wr := NewWriter(buf, Int16DataType, []int32{1, 2, 2})

	sample := []int16{1, 2, 3, 4}
	err := wr.WriteInt16(sample)
	if err != nil {
		t.Fatal(err)
	}

	b := buf.Bytes()

	header := b[0:16]
	expectedHeader := []byte{
		0x00, 0x00, 0x0B, 0x03,
		0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x00, 0x00, 0x02,
	}
	if !bytes.Equal(header, expectedHeader) {
		t.Fatal("Wrote unexpected header", header, expectedHeader)
	}

	el := b[16:]
	if !bytes.Equal(el, []byte{
		0x00, 0x01,
		0x00, 0x02,
		0x00, 0x03,
		0x00, 0x04,
	}) {
		t.Fatal("Wrote unexpected elements", el)
	}
}

func TestWriteInt32(t *testing.T) {
	buf := new(bytes.Buffer)
	wr := NewWriter(buf, Int32DataType, []int32{1, 2, 2})

	sample := []int32{1, 2, 3, 4}
	err := wr.WriteInt32(sample)
	if err != nil {
		t.Fatal(err)
	}

	b := buf.Bytes()

	header := b[0:16]
	expectedHeader := []byte{
		0x00, 0x00, 0x0C, 0x03,
		0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x00, 0x00, 0x02,
	}
	if !bytes.Equal(header, expectedHeader) {
		t.Fatal("Wrote unexpected header", header, expectedHeader)
	}

	el := b[16:]
	if !bytes.Equal(el, []byte{
		0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x00, 0x00, 0x03,
		0x00, 0x00, 0x00, 0x04,
	}) {
		t.Fatal("Wrote unexpected elements", el)
	}
}

func TestWriteFloat32(t *testing.T) {
	buf := new(bytes.Buffer)
	wr := NewWriter(buf, Float32DataType, []int32{1, 2, 2})

	sample := []float32{1, 2, 3, 4}
	err := wr.WriteFloat32(sample)
	if err != nil {
		t.Fatal(err)
	}

	b := buf.Bytes()

	header := b[0:16]
	expectedHeader := []byte{
		0x00, 0x00, 0x0D, 0x03,
		0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x00, 0x00, 0x02,
	}
	if !bytes.Equal(header, expectedHeader) {
		t.Fatal("Wrote unexpected header", header, expectedHeader)
	}

	el := b[16:]
	if !bytes.Equal(el, []byte{
		0x3f, 0x80, 0x00, 0x00,
		0x40, 0x00, 0x00, 0x00,
		0x40, 0x40, 0x00, 0x00,
		0x40, 0x80, 0x00, 0x00,
	}) {
		t.Fatal("Wrote unexpected elements", el)
	}
}

func TestWriteFloat64(t *testing.T) {
	buf := new(bytes.Buffer)
	wr := NewWriter(buf, Float64DataType, []int32{1, 2, 2})

	sample := []float64{1, 2, 3, 4}
	err := wr.WriteFloat64(sample)
	if err != nil {
		t.Fatal(err)
	}

	b := buf.Bytes()

	header := b[0:16]
	expectedHeader := []byte{
		0x00, 0x00, 0x0E, 0x03,
		0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x00, 0x00, 0x02,
	}
	if !bytes.Equal(header, expectedHeader) {
		t.Fatal("Wrote unexpected header", header, expectedHeader)
	}

	el := b[16:]
	if !bytes.Equal(el, []byte{
		0x3f, 0xf0, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x40, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x40, 0x08, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x40, 0x10, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
	}) {
		t.Fatal("Wrote unexpected elements", el)
	}
}
