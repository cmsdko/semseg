# --- Build Stage ---
FROM golang:1.21-alpine AS builder

# Set the working directory
WORKDIR /app

# Copy the go.mod and go.sum files for both the library and the example
# This allows Docker to cache dependencies efficiently
COPY go.mod go.sum ./
COPY example/go.mod example/go.sum ./example/

# The 'replace' directive in example/go.mod will now work because the parent go.mod is present
# We run `go mod download` from the root to ensure all dependencies are fetched.
RUN go mod download

# Copy the entire source code
# The library source code will be at /app/
# The example source code will be at /app/example/
COPY . .

# Build the binary FROM the 'example' directory
# The output will be placed at /app/server
RUN CGO_ENABLED=0 go build -o /app/server ./example

# --- Final Stage ---
FROM alpine:latest
WORKDIR /root/

# Copy the compiled binary from the builder stage
COPY --from=builder /app/server .

EXPOSE 8080
CMD ["./server"]