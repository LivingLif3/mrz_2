// package main

// import (
// 	"fmt"
// )

// type HopfieldNetwork struct {
// 	weights [][]float64
// 	size    int
// 	trainingSet [][]int
// }

// // NewHopfieldNetwork initializes a new Hopfield Network
// func NewHopfieldNetwork(alphabet [][]int) *HopfieldNetwork {
// 	size := len(alphabet[0])
// 	weights := make([][]float64, size)
// 	for i := range weights {
// 		weights[i] = make([]float64, size)
// 	}

// 	network := &HopfieldNetwork{
// 		weights: weights,
// 		size:    size,
// 		trainingSet: alphabet,
// 	}

// 	network.train(alphabet)
// 	return network
// }

// // train adjusts the weights using the delta projection rule
// func (hn *HopfieldNetwork) train(alphabet [][]int) {
// 	for _, image := range alphabet {
// 		for i := 0; i < hn.size; i++ {
// 			for j := 0; j < hn.size; j++ {
// 				if i != j {
// 					hn.weights[i][j] += float64(image[i] * image[j])
// 				}
// 			}
// 		}
// 	}
// }

// // predict attempts to find the stable state of the network
// func (hn *HopfieldNetwork) predict(input []int) ([]int, bool) {
// 	state := make([]int, len(input))
// 	copy(state, input)

// 	for {
// 		updated := false
// 		for i := 0; i < hn.size; i++ {
// 			sum := 0.0
// 			for j := 0; j < hn.size; j++ {
// 				sum += hn.weights[i][j] * float64(state[j])
// 			}
// 			newState := sign(sum)
// 			if state[i] != newState {
// 				state[i] = newState
// 				updated = true
// 			}
// 		}
// 		if !updated {
// 			break
// 		}
// 	}

// 	for _, image := range hn.trainingSet {
// 		if equal(state, image) || equal(state, negate(image)) {
// 			return state, true
// 		}
// 	}

// 	return state, false
// }

// // sign function to determine the state
// func sign(x float64) int {
// 	if x > 0 {
// 		return 1
// 	} else if x < 0 {
// 		return -1
// 	}
// 	return 0
// }

// // equal checks if two slices are equal
// func equal(a, b []int) bool {
// 	if len(a) != len(b) {
// 		return false
// 	}
// 	for i := range a {
// 		if a[i] != b[i] {
// 			return false
// 		}
// 	}
// 	return true
// }

// // negate inverts a binary pattern
// func negate(a []int) []int {
// 	neg := make([]int, len(a))
// 	for i := range a {
// 		neg[i] = -a[i]
// 	}
// 	return neg
// }

// // printImage prints the image in a readable format
// func printImage(image []int, width int) {
// 	for i, val := range image {
// 		if i > 0 && i%width == 0 {
// 			fmt.Println()
// 		}
// 		if val == 1 {
// 			fmt.Print("⬜")
// 		} else {
// 			fmt.Print("⬛")
// 		}
// 	}
// 	fmt.Println()
// }

// func main() {
// 	alphabet := [][]int{
// 		{-1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1}, // Распакованная первая матрица
// 		{1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1},  // Распакованная вторая матрица
// 		{-1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1}, // Третья матрица
// 		{-1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1}, // Четвертая матрица
// 	}

// 	network := NewHopfieldNetwork(alphabet)

// 	// Тестовая картинка - может быть изменена
// 	testImage := []int{1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1}

// 	result, recognized := network.predict(testImage)
// 	if recognized {
// 		fmt.Println("Recognized pattern:")
// 		printImage(result, 4)
// 	} else {
// 		fmt.Println("Pattern not recognized")
// 	}
// }

