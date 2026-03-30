#!/usr/bin/env bash
# Install Neo4j 5 Community on JarvisLabs (no Docker)
set -euo pipefail
NEO4J_VERSION="5.18.0"
INSTALL_DIR="/home/neo4j"

echo "Installing Neo4j $NEO4J_VERSION to $INSTALL_DIR…"
curl -L "https://dist.neo4j.org/neo4j-community-${NEO4J_VERSION}-unix.tar.gz" \
    | tar -xz -C /home/
mv "/home/neo4j-community-${NEO4J_VERSION}" "$INSTALL_DIR" 2>/dev/null || true

# Set password from .env
source "$(dirname "$0")/../../../.env" 2>/dev/null || true
NEO4J_PASSWORD="${NEO4J_PASSWORD:-changeme}"

"$INSTALL_DIR/bin/neo4j-admin" dbms set-initial-password "$NEO4J_PASSWORD"
echo "export NEO4J_HOME=$INSTALL_DIR" >> ~/.bashrc
echo "export PATH=\$NEO4J_HOME/bin:\$PATH" >> ~/.bashrc
echo "Neo4j installed at $INSTALL_DIR"
echo "Start with: $INSTALL_DIR/bin/neo4j start"
