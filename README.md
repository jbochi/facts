# facts

[![Build Status](https://travis-ci.org/jbochi/facts.svg?branch=master)](https://travis-ci.org/jbochi/facts)

Matrix Factorization based recommender system in Go. Because **facts** are more important than ever.

This project provides a `vectormodel` package that can be used to serve real time recommendations. First of all, you will need to train a model to get document embeddings or latent **fact**ors. I highly recommend the [implicit](https://github.com/benfred/implicit) library for that. Once you have the documents as a map of `int` ids to arrays of `float64`, you can create the vector model by calling:

`model, err := NewVectorModel(documents map[int][]float64, confidence, regularization float64)`

And to generate recommendations call `.Recommend` with a set of items the user has seen:

`recs := model.Recommend(seenDocs *map[int]bool, n int)`

Note that user vectors are not required. Matter of fact, you can use this to recommend documents to users that were *not* in the training set. The recommendations will be computed very efficiently (probably <1ms, depends on your model size) in real time.

Check out the [demo](https://github-recs.appspot.com/) for a complete example that recommends GitHub repositories.

Demo source code is available here: https://github.com/jbochi/github-recs
