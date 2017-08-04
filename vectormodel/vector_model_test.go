package vectormodel

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestVectorModelConstructor(t *testing.T) {
	confidence := 1.0
	regularization := 0.01
	docs := make(map[int][]float64)
	docs[1234] = []float64{1, 2, 3}
	vm, err := NewVectorModel(docs, confidence, regularization)

	if err != nil {
		t.Fatalf("Failed to create vector model %s", err)
	}

	if vm.nFactors != 3 {
		t.Errorf("Expecting 3 factors, got %d", vm.nFactors)
	}
	if vm.confidence != confidence {
		t.Errorf("Wrong confidence: %f", vm.confidence)
	}
	if vm.regularization != regularization {
		t.Errorf("Wrong regularization: %f", vm.regularization)
	}

	Y := vm.itemFactorsY
	YtY := vm.squaredItemFactorsYtY
	Yrows, Ycols := Y.Dims()
	if Yrows != 1 || Ycols != 3 {
		t.Errorf("Y has wrong dimensions (%d, %d) instead of (3, 1)", Yrows, Ycols)
	}

	a, b, c := Y.At(0, 0), Y.At(0, 1), Y.At(0, 2)
	if a != 1 || b != 2 || c != 3 {
		t.Errorf("Y has the wrong values %f, %f, %f", a, b, c)
	}

	YtYrows, YtYcols := YtY.Dims()
	if YtYrows != 3 || YtYcols != 3 {
		t.Errorf("YtY has wrong dimensions (%d, %d) instead of (3, 3)",
			YtYrows, YtYcols)
	}

	a, b, c = YtY.At(0, 0), YtY.At(0, 1), YtY.At(0, 2)
	if a != 1 || b != 2 || c != 3 {
		t.Errorf("first row of YtY has the wrong values: %f, %f, %f", a, b, c)
	}
	a, b, c = YtY.At(1, 0), YtY.At(1, 1), YtY.At(1, 2)
	if a != 2 || b != 4 || c != 6 {
		t.Errorf("second row of YtY has the wrong values: %f, %f, %f", a, b, c)
	}
	a, b, c = YtY.At(2, 0), YtY.At(2, 1), YtY.At(2, 2)
	if a != 3 || b != 6 || c != 9 {
		t.Errorf("third row of YtY has the wrong values: %f, %f, %f", a, b, c)
	}

}

func TestVectorModelConstructorWithInvalidVectors(t *testing.T) {
	docs := make(map[int][]float64)
	docs[1234] = []float64{1, 2, 3}
	docs[1235] = []float64{1, 2, 3, 5}
	_, err := NewVectorModel(docs, 1.0, 0.01)

	if err == nil {
		t.Fatalf("Should not allow vectors with different sizes")
	}
}

func TestUserVector(t *testing.T) {
	defaultConfidence := 40.0
	regularization := 0.01
	docs := make(map[int][]float64)
	docs[1234] = []float64{1, 2, 3}
	vm, err := NewVectorModel(docs, defaultConfidence, regularization)
	if err != nil {
		t.Fatalf("Failed to create vector model %s", err)
	}

	confidence := map[int]float64{1234: 40.0, 666: 1.0}

	user, err := vm.userVector(confidence)

	if err != nil {
		t.Fatalf("Error solving user vector: %s", err)
	}

	rows, cols := user.Dims()
	if rows != 3 || cols != 1 {
		t.Fatalf("Invalid user vec dimensions: %d, %d", rows, cols)
	}

	a, b, c := user.At(0, 0), user.At(1, 0), user.At(2, 0)
	if math.Abs(a-0.0714273)+math.Abs(b-0.14285459)+math.Abs(c-0.21428189) > 1e-4 {
		t.Fatalf("Invalid user vec: [%f, %f, %f]", a, b, c)
	}
}

func BenchmarkUserVector(b *testing.B) {
	var err error

	defaultConfidence := 40.0
	regularization := 0.01
	docs := make(map[int][]float64)
	docs[1234] = []float64{1, 2, 3}
	vm, err := NewVectorModel(docs, defaultConfidence, regularization)
	if err != nil {
		b.Fatalf("Failed to create vector model %s", err)
	}

	confidence := map[int]float64{1234: 40.0, 666: 1.0}

	var user mat.VecDense

	// Reset benchmark timer
	b.ResetTimer()

	// Run benchmark
	for i := 0; i < b.N; i++ {
		user, err = vm.userVector(confidence)
	}
	if err != nil {
		b.Fatalf("Error solving user vector: %s", err)
	}

	rows, cols := user.Dims()
	if rows != 3 || cols != 1 {
		b.Fatalf("Invalid user vec dimensions: %d, %d", rows, cols)
	}

	x, y, z := user.At(0, 0), user.At(1, 0), user.At(2, 0)
	if math.Abs(x-0.0714273)+math.Abs(y-0.14285459)+math.Abs(z-0.21428189) > 1e-4 {
		b.Fatalf("Invalid user vec: [%f, %f, %f]", x, y, z)
	}
}

func TestScoresForUserVec(t *testing.T) {
	regularization := 0.01
	confidence := 40.0
	docs := make(map[int][]float64)
	docs[1234] = []float64{1, 2, 3}
	docs[4567] = []float64{3, 2, 1}
	vm, err := NewVectorModel(docs, confidence, regularization)
	if err != nil {
		t.Fatalf("Failed to create vector model %s", err)
	}

	userVec := mat.NewVecDense(3, []float64{0.2, 0.1, 0.0})
	scores := vm.scoresForUserVec(userVec)

	rows, cols := scores.Dims()
	if rows != 2 || cols != 1 {
		t.Fatalf("Invalid scores dimensions: %d, %d", rows, cols)
	}

	score1 := scores.At(vm.docIndexes[1234], 0)
	score2 := scores.At(vm.docIndexes[4567], 0)
	if score1 != (0.2*1+0.1*2) || score2 != (3*0.2+2*0.1) {
		t.Fatalf("Invalid scores: %f (%d), %f (%d)", score1, vm.docIndexes[1234], score2, vm.docIndexes[4567])
	}
}

func TestRecommend(t *testing.T) {
	confidence := 40.0
	regularization := 0.01
	docs := make(map[int][]float64)
	docs[1234] = []float64{1, 2, 3}
	docs[4567] = []float64{3, 2, 1}
	vm, err := NewVectorModel(docs, confidence, regularization)
	if err != nil {
		t.Fatalf("Failed to create vector model %s", err)
	}

	seenDocs := map[int]bool{1234: true}
	n := 10

	recommendations, err := vm.Recommend(&seenDocs, n)
	if err != nil {
		t.Fatalf("Failed to recommend %s", err)
	}
	if len(recommendations) != 2 {
		t.Fatalf("Wrong number of recommendations: %v", recommendations)
	}
	if recommendations[0].DocumentID != 4567 {
		t.Errorf("Wrong recommendation: %v", recommendations[0])
	}
	if recommendations[1].DocumentID != 1234 {
		t.Errorf("Wrong recommendation: %v", recommendations[1])
	}

	// This is how you can obtain the excpected scores in python:
	//
	// import numpy as np
	// Y = np.array([[1, 2, 3], [3, 2, 1]])
	// YtY = Y.T.dot(Y)
	// YtY
	// regularization = 0.01
	// confidence = 40
	// A = YtY + regularization * np.eye(3)
	// b = np.zeros(3)
	// factor = Y[0]
	// A += (confidence - 1) * np.outer(factor, factor)
	// b += confidence * factor
	// user = np.linalg.solve(A, b)
	// np.dot(Y, user)

	if math.Abs(recommendations[0].Score-0.00104011) > 1e-5 {
		t.Errorf("Wrong score: %f", recommendations[0].Score)
	}
}

func TestRecommendReturnsTopItems(t *testing.T) {
	confidence := 40.0
	regularization := 0.01
	docs := make(map[int][]float64)
	docs[0] = []float64{1, 2, 3}
	docs[1] = []float64{1, 2, 3.01}
	docs[2] = []float64{1, 2, 3.02}
	docs[3] = []float64{3, 2, 1}
	docs[4] = []float64{1, 2, 3.03}
	vm, err := NewVectorModel(docs, confidence, regularization)
	if err != nil {
		t.Fatalf("Failed to create vector model %s", err)
	}

	seenDocs := map[int]bool{0: true}
	n := 3
	recs, err := vm.Recommend(&seenDocs, n)
	if err != nil {
		t.Fatalf("Failed to recommend %s", err)
	}
	if len(recs) != 3 {
		t.Fatalf("Wrong number of recommendations: %v", recs)
	}
	if recs[0].DocumentID != 1 || recs[1].DocumentID != 2 || recs[2].DocumentID != 4 {
		t.Errorf("Wrong recommendations: %v", recs)
	}
}

func TestRankSortsTopItems(t *testing.T) {
	confidence := 40.0
	regularization := 0.01
	docs := make(map[int][]float64)
	docs[0] = []float64{1, 2, 3}
	docs[1] = []float64{1, 2, 3}
	docs[2] = []float64{1, 2, 3}
	docs[3] = []float64{3, 2, 1}
	docs[4] = []float64{1, 2, 3}
	vm, err := NewVectorModel(docs, confidence, regularization)
	if err != nil {
		t.Fatalf("Failed to create vector model %s", err)
	}

	seenDocs := map[int]bool{0: true}
	items := []int{0, 1, 3, 10}
	err = vm.Rank(&items, &seenDocs)
	if err != nil {
		t.Fatalf("Failed to recommend %s", err)
	}
	// Order is: Most similar, Not read, Unknown, read
	if items[0] != 1 || items[1] != 3 || items[2] != 10 || items[3] != 0 {
		t.Errorf("Wrong recommendations: %v", items)
	}
}
