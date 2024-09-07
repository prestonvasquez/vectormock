package main

import (
	"context"
	"log"
	"os"

	"github.com/prestonvasquez/vectormock"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores/mongovector"
	"go.mongodb.org/mongo-driver/v2/mongo"
	"go.mongodb.org/mongo-driver/v2/mongo/options"
)

// This test requires the URI to point to a MongoDB Atlas Cluster. It also
// requires that a "langchaingo-test" and "vstore" database and collection exist
// with the following vector search index:
//
//{
//  "fields": [{
//    "type": "vector",
//    "path": "plot_embedding",
//    "numDimensions": 1536,
//    "similarity": "dotProduct"
//  }]
//}
//
// For more information on MongoDB Vector Databases, see this tutorial:
// https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/vector-search-quick-start/

const (
	testDB          = "langchaingo-test"
	testColl        = "vstore"
	testIndexDP1536 = "vector_index_dotProduct_1536"
)

func main() {
	uri := os.Getenv("MONGODB_URI")
	if uri == "" {
		log.Fatal("MONGODB_URI must be set")
	}

	// Connect to MongoDB.
	client, err := mongo.Connect(options.Client().ApplyURI(uri))
	if err != nil {
		log.Fatalf("failed to connect to MongoDB: %v", err)
	}

	coll := client.Database(testDB).Collection(testColl)

	// Create the mock emedder.
	emb := vectormock.NewDotProduct(1536)

	mockDocs := []vectormock.Document{
		{PageContent: "Gabriel García Márquez", Score: 0.80},
		{PageContent: "Gabriela Mistral", Score: 0.67},
		{PageContent: "Miguel de Cervantes", Score: 0.09},
	}

	emb.MockDocuments(mockDocs...)

	// Use LangChainGo to store the vectors in MongoDB. You do not need to use
	// LangChainGo to mock an embedding, this is just a conveniecne for the sake
	// of this example.
	store := mongovector.New(*coll, emb, mongovector.WithIndex(testIndexDP1536))

	// conver mockDocs to schema.Document
	schemaDocs := make([]schema.Document, len(mockDocs))
	for i := range mockDocs {
		schemaDocs[i] = schema.Document{
			PageContent: mockDocs[i].PageContent,
			Score:       mockDocs[i].Score,
		}
	}

	_, err = store.AddDocuments(context.Background(), schemaDocs)

	// Perform a simlarity search. Note that the actual query doesn't matter at
	// all. The mock handles returning the embedded query vector.
	results, err := store.SimilaritySearch(context.Background(), "Latin Authors", 10)

	for _, res := range results {
		log.Printf("PageContent: %s, Score: %.2f", res.PageContent, res.Score)
	}
}
