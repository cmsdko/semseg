# --- Build Stage ---
# Use the official Golang Alpine image as a builder
FROM golang:1.21-alpine AS builder

# Set the working directory inside the container
WORKDIR /app

# To leverage Docker layer caching, we'll copy module files first
# and download dependencies before copying the source code.
#COPY go.mod  ./
COPY example/go.mod example/go.sum ./example/
#RUN --mount=type=cache,target=/go/pkg/mod go mod download

# Copy the entire project source code
COPY . .

# Switch to the example directory, where we'll modify go.mod for local development
WORKDIR /app/example

# This build argument allows toggling the local replace directive.
# It can be set from docker-compose.yml or with `docker build --build-arg USE_LOCAL=1`.
ARG USE_LOCAL
RUN if [ "$USE_LOCAL" = "1" ]; then \
      printf '\nreplace github.com/cmsdko/semseg => ../\n' >> go.mod && go mod tidy; \
    fi

# Build the application binary.
# CGO_ENABLED=0 creates a static binary, which is crucial for the final alpine image.
RUN CGO_ENABLED=0 go build -o /out/example .

# --- Final Stage ---
# Use a minimal alpine image for a small final image size
FROM alpine:latest

# Set a non-root working directory as a best practice
WORKDIR /app

# Copy the compiled binary from the builder stage
COPY --from=builder /out/example .

# Expose the port the application runs on
EXPOSE 8080

# The command to run when the container starts
CMD ["./example"]
