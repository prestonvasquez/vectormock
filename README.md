# vectormock

This library is for mocking an embedding model based on similarity scores. The idea is to embed data with a relative similarity score to a root query vector. 

```go
const openAiAda002Dim = 1536
emb := mockvector.NewDotProduct(openAiAda002Dim) // query vector generated under the hood

emb.MockDocuments(
  	vectormock.Document{PageContent: "Gabriel García Márquez", Score: 0.80},
	vectormock.Document{PageContent: "Gabriela Mistral", Score: 0.67},
	vectormock.Document{PageContent: "Miguel de Cervantes", Score: 0.09})

results, _ := store.SimilaritySearch(context.Background(), "Latin Authors", 3)

for _, res := range results {
	log.Printf("PageContent: %s, Score: %.2f", res.PageContent, res.Score)
}

// Output: 
// 2024/09/06 22:33:48 PageContent: Gabriel García Márquez, Score: 0.80
// 2024/09/06 22:33:48 PageContent: Gabriela Mistral, Score: 0.67
// 2024/09/06 22:33:48 PageContent: Miguel de Cervantes, Score: 0.09
```

For full example see [here](examples/mongodb/main.go).

