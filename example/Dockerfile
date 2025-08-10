# --- Build Stage ---
FROM golang:1.21-alpine AS builder

# Set the working directory
WORKDIR /app

# Copy the go.mod and go.sum files for both the library and the example
# This allows Docker to cache dependencies efficiently
COPY go.mod go.sum ./

# The 'replace' directive in example/go.mod will now work because the parent go.mod is present
# We run `go mod download` from the root to ensure all dependencies are fetched.
RUN go mod download

COPY . .
RUN go mod tidy
RUN CGO_ENABLED=0 go build -o /app/example .

# --- Final Stage ---
FROM alpine:latest
WORKDIR /root/

# Copy the compiled binary from the builder stage
COPY --from=builder /app/example .

EXPOSE 8080
CMD ["./example"]
