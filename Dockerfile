FROM golang:1.22.3

WORKDIR /app

# Copy Go modules and install dependencies
COPY go.mod go.sum ./
RUN go mod tidy

# Copy the rest of the application code
COPY . .

# Install necessary tools for Python and create a virtual environment
RUN apt-get update && apt-get install -y python3-venv python3-dev build-essential && \
    python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip

# Set the PATH to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY lib/requirements.txt /app/lib/requirements.txt

RUN /opt/venv/bin/pip install -r /app/lib/requirements.txt

# Build the Go application
RUN go build -o bin/main cmd/main.go

EXPOSE 8080

ENTRYPOINT ["./bin/main"]
