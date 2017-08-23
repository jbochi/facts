# facts
Matrix Factorization based recommender system in Go. Because **facts** are more important than ever.

This project provides a `vectormodel` package that can be used to serve real time recommendations. First of all, you will need to train a model to get document embeddings or latent **fact**ors. I highly recommend the [implicit](https://github.com/benfred/implicit) library for that. Once you have the documents as a map of `int` ids to arrays of `float64`, you can create the vector model by calling:

`model, err := NewVectorModel(documents map[int][]float64, confidence, regularization float64)`

And to generate recommendations, you do:

`recs := model.Recommend(seenDocs *map[int]bool, n int)`


Check out the [demo](https://github-recs.appspot.com/) for a complete example that recommends GitHub repositories.

Demo source code is available here: https://github.com/jbochi/github-recs
