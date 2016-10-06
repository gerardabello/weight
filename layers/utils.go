package layers

import "math/rand"

const letterBytes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

func RandomID(n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = letterBytes[rand.Int63()%int64(len(letterBytes))]
	}
	return string(b)
}

func InterfaceArrayToIntArray(ia interface{}) ([]int, bool) {
	var ok bool

	a, ok := ia.([]interface{})
	if !ok {
		return nil, false
	}

	b := make([]int, len(a))
	for i := range a {
		var v float64
		v, ok = a[i].(float64)

		if !ok {
			return nil, false
		}

		b[i] = int(v)
	}

	return b, true
}

func InterfaceFloatToInt(i interface{}) (int, bool) {
	var ok bool
	var fv float64

	fv, ok = i.(float64)

	if !ok {
		return 0, false
	}

	return int(fv), true
}
