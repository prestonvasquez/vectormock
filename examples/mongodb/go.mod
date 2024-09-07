module github.com/prestonvasquez/vectormock/examples/mongodb

go 1.23

replace github.com/prestonvasquez/vectormock => ../../

replace github.com/tmc/langchaingo => github.com/prestonvasquez/langchaingo v0.0.0-20240906225419-daff9150b5cf

require (
	github.com/prestonvasquez/vectormock v0.0.0-00010101000000-000000000000
	github.com/tmc/langchaingo v0.0.0-00010101000000-000000000000
	go.mongodb.org/mongo-driver/v2 v2.0.0-beta1
)

require (
	github.com/dlclark/regexp2 v1.10.0 // indirect
	github.com/golang/snappy v0.0.4 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/klauspost/compress v1.17.6 // indirect
	github.com/pkoukk/tiktoken-go v0.1.6 // indirect
	github.com/xdg-go/pbkdf2 v1.0.0 // indirect
	github.com/xdg-go/scram v1.1.2 // indirect
	github.com/xdg-go/stringprep v1.0.4 // indirect
	github.com/youmark/pkcs8 v0.0.0-20181117223130-1be2e3e5546d // indirect
	golang.org/x/crypto v0.23.0 // indirect
	golang.org/x/sync v0.7.0 // indirect
	golang.org/x/text v0.15.0 // indirect
)