// Package vectormodel provides primitives for serving Matrix Factorization
// based Recommender System models.
// We assume that the document vectors were calculated according to the paper
// "Collaborative Filtering for Implicit Feedback Datasets".
// Given the document vectors and the list of documents consumed by a user,
// this package can calculate the user vectors in realtime to generate
// recommendations.
package vectormodel

import (
	"errors"
	"fmt"
	"sort"

	"gonum.org/v1/gonum/mat"
)

type (
	// VectorModel is a struct to handle document vector space models.
	VectorModel struct {
		confidence            float64
		regularization        float64
		docIDs                []int
		docIndexes            map[int]int
		nFactors              int
		itemFactorsY          *mat.Dense
		squaredItemFactorsYtY *mat.Dense
	}

	// DocumentScore is the result of a recommendation
	DocumentScore struct {
		DocumentID int
		Score      float64
	}
)

func (a byDocScoreDesc) Len() int           { return len(a) }
func (a byDocScoreDesc) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byDocScoreDesc) Less(i, j int) bool { return a[i].Score > a[j].Score }

type byDocScoreDesc []DocumentScore

// NewVectorModel creates a new VectorModel
func NewVectorModel(documents map[int][]float64, confidence, regularization float64) (*VectorModel, error) {
	var vm VectorModel
	vm.confidence = confidence
	vm.regularization = regularization
	vm.docIDs = make([]int, len(documents))
	vm.docIndexes = make(map[int]int)

	data := make([]float64, 0)
	i := 0
	for doc, vector := range documents {
		if i == 0 {
			vm.nFactors = len(vector)
		} else if len(vector) != vm.nFactors {
			return nil, errors.New("Invalid vector size")
		}
		vm.docIndexes[doc] = i
		vm.docIDs[i] = doc
		data = append(data, vector...)
		i++
	}
	vm.itemFactorsY = mat.NewDense(len(documents), vm.nFactors, data)

	var YtY mat.Dense
	YtY.Mul(vm.itemFactorsY.T(), vm.itemFactorsY)
	vm.squaredItemFactorsYtY = &YtY

	return &vm, nil
}

// Rank sorts a list of candidate assets for a given user history
func (vm *VectorModel) Rank(candidates *[]int, seenDocs *map[int]bool) (scores []float64, err error) {
	candidateScores, err := vm.scoreCandidates(candidates, seenDocs)
	if err != nil {
		return nil, err
	}
	scores = make([]float64, len(candidateScores))
	for i, candidateScore := range candidateScores {
		(*candidates)[i] = candidateScore.DocumentID
		scores[i] = candidateScore.Score
	}
	return scores, nil
}

// Recommend returns a list of recommendedDocs and a list of scores
func (vm *VectorModel) Recommend(seenDocs *map[int]bool, n int) (recommendations []DocumentScore, err error) {
	recommendations, err = vm.scoreCandidates(&vm.docIDs, seenDocs)
	if err != nil {
		return nil, err
	}
	if len(recommendations) > n {
		recommendations = recommendations[:n]
	}
	return recommendations, nil
}

func (vm *VectorModel) scoreCandidates(candidates *[]int, seenDocs *map[int]bool) (recommendations []DocumentScore, err error) {
	confidenceMap := vm.confidenceMap(seenDocs)
	if len(confidenceMap) == 0 {
		return nil, fmt.Errorf("No seen doc is in model. History: %d Model: %d",
			len(*seenDocs), len(vm.docIndexes))
	}
	userVec, err := vm.userVector(confidenceMap)
	if err != nil {
		return recommendations, err
	}
	scoresVec := vm.scoresForUserVec(&userVec)
	candidateScores := make([]DocumentScore, len(*candidates))
	for i, doc := range *candidates {
		var score float64
		if _, docAlreadySeen := (*seenDocs)[doc]; docAlreadySeen {
			score = -1
		} else if docIndex, docInModel := vm.docIndexes[doc]; !docInModel {
			score = 0
		} else {
			score = scoresVec.At(docIndex, 0)
		}
		candidateScores[i] = DocumentScore{doc, score}
	}
	sort.Sort(byDocScoreDesc(candidateScores))
	return candidateScores, nil
}

func (vm *VectorModel) confidenceMap(seenDocs *map[int]bool) map[int]float64 {
	confidenceMap := make(map[int]float64)
	for doc := range *seenDocs {
		if _, inModel := vm.docIndexes[doc]; inModel {
			confidenceMap[doc] = vm.confidence
		}
	}
	return confidenceMap
}

// userVector returns the user vector for a given set of consumed documents
func (vm *VectorModel) userVector(confidenceMap map[int]float64) (mat.VecDense, error) {
	// We follow the notation from the paper "Collaborative Filtering for Implicit Feedback Datasets"
	// Please see github.com/benfred/implicit as a reference implementation

	// We solve the following linear equation:
	// Xu = (YtCuY + regularization*I)i^-1 * YtYCuPu

	// A = YtCuY + reg * I = YtY + reg * I + Yt(Cu - I)Y
	// We initialize A to YtY + reg * I  and sum the last term for each doc
	var A mat.Dense
	A.Add(vm.squaredItemFactorsYtY, eye(vm.nFactors, vm.regularization))

	// b = YtCuPu
	b := mat.NewVecDense(vm.nFactors, make([]float64, vm.nFactors))

	for doc, confidence := range confidenceMap {
		index, docFound := vm.docIndexes[doc]
		if !docFound {
			continue
		}
		factor := vm.itemFactorsY.RowView(index)

		// A += (confidence - 1) * np.outer(factor, factor)
		var factor2 mat.Dense
		factor2.Mul(factor, factor.T())
		factor2.Scale(confidence-1, &factor2)
		A.Add(&A, &factor2)

		// b += confidence * factor
		b.AddScaledVec(b, confidence, factor)
	}

	var x mat.VecDense
	// We could just solve the matrix by calling the next line, but
	// A is positively defined, so we can use the Cholesky solver
	// err := x.SolveVec(&A, b)

	var ch mat.Cholesky
	if ok := ch.Factorize(&unsafeSymmetric{A, vm.nFactors}); !ok {
		return x, errors.New("Failed to run Cholesky factorization")
	}
	err := ch.SolveVec(&x, b)
	return x, err
}

// scoresForUserVec returns a vector with scores given set of consumed documents
func (vm *VectorModel) scoresForUserVec(userVec *mat.VecDense) mat.VecDense {
	var y mat.VecDense
	y.MulVec(vm.itemFactorsY, userVec)
	return y
}

// eye returns an identity matrix with size n and `value` in the diagonal
func eye(n int, value float64) mat.Matrix {
	m := mat.NewDense(n, n, make([]float64, n*n))
	for i := 0; i < n; i++ {
		m.Set(i, i, value)
	}
	return m
}

type unsafeSymmetric struct {
	mat.Dense
	n int
}

func (s *unsafeSymmetric) Symmetric() int {
	return s.n
}
