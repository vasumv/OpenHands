# Multi-stage build to get Fray from the official Fray image
FROM ghcr.io/cmu-pasta/fray:0.6.9 AS fray-source

# Start from pre-built OpenHands runtime image
# This already has all OpenHands dependencies, Python, Node.js, bash, apt-get, etc.
FROM ghcr.io/openhands/runtime:0.62-nikolaik

# Install Java and Nix (required for Gradle/Fray benchmarks)
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-21-jdk \
        curl \
        xz-utils && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for Java 21 (Fray requires Java 21)
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Copy Nix and Fray installation from the official Fray image
# The Fray image already has Nix and Fray installed
COPY --from=fray-source /nix /nix

# Set up Nix environment
ENV PATH="/nix/var/nix/profiles/default/bin:/root/.nix-profile/bin:$PATH"
ENV NIX_PATH="/nix/var/nix/profiles/per-user/root/channels"

# Initialize Nix for the root user
RUN mkdir -p /root/.nix-defexpr && \
    mkdir -p /nix/var/nix/profiles/per-user/root && \
    ln -sf /nix/var/nix/profiles/default /root/.nix-profile || true

# Clone fray-benchmark repository
WORKDIR /workspace
RUN git clone https://github.com/cmu-pasta/fray-benchmark.git && \
    chown -R openhands:openhands /workspace/fray-benchmark

# Set the working directory
WORKDIR /workspace/fray-benchmark

# Switch back to openhands user
USER openhands
