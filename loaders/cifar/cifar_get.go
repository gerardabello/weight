package cifar

import (
	"encoding/binary"
	"errors"
	"reflect"
)

func (m *CIFAR) GetNextRow(row interface{}) error {
	m.mutex.Lock()
	file := m.batches[m.currentBatch]

	currentOffset, err := file.Seek(0, 1) //moves 0 bytes relative to the current position (= no movement) and returns the current position
	if err != nil {
		return err
	}

	rowSize := m.rowType.Size()

	if currentOffset%int64(rowSize) != 0 {
		panic("The current offset in the reading is not a multiple of the row size in bytes")
	}

	if reflect.Indirect(reflect.ValueOf(row)).Type().Size() != rowSize {
		panic("Row object supplied has an unexpected type")
	}

	currentRow := currentOffset / int64(rowSize)

	if currentRow == int64(m.batchRows[m.currentBatch]) {
		m.currentBatch++
		if m.currentBatch >= len(m.batches) {
			return errors.New("No new set available")
		}
		m.batches[m.currentBatch].Seek(0, 0) //set net batch to start

		m.mutex.Unlock()
		//Try again
		return m.GetNextRow(row)
	}

	err = binary.Read(file, binary.LittleEndian, row)
	if err != nil {
		return err
	}

	m.mutex.Unlock()
	return nil
}
