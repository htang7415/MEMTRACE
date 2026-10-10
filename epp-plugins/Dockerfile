# llm-d's EPP with MEMTRACE's plugins (cmd/epp), built from this module; the llm-d-router version is go.mod's.
# Mirrors upstream's Dockerfile.epp: static binary, distroless runtime, same ports and user.
ARG BASE_IMAGE=gcr.io/distroless/static:nonroot

FROM --platform=${BUILDPLATFORM} golang:1.26.8 AS go-builder
ARG TARGETOS
ARG TARGETARCH
WORKDIR /workspace
COPY go.mod go.sum ./
RUN go mod download
COPY capacityload/ capacityload/
COPY cmd/ cmd/
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -ldflags="-s -w" -o bin/epp ./cmd/epp

FROM ${BASE_IMAGE}
WORKDIR /
COPY --from=go-builder /workspace/bin/epp /app/epp
USER 65532:65532
# gRPC, health, metrics, KV-events ZMQ SUB socket
EXPOSE 9002 9003 9090 5557
ENTRYPOINT ["/app/epp"]
