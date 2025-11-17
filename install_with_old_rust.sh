#!/bin/bash
# Install dependencies using Rust 1.73.0 with lint override for tokenizers compatibility

set -e

echo "=========================================="
echo "Installing with Rust 1.73.0 for tokenizers compatibility"
echo "=========================================="
echo ""

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo "Rust is not installed. Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Install Rust 1.73.0 (supports edition2024 and can compile tokenizers 0.13.3 with lint override)
echo "Installing Rust 1.73.0..."
rustup install 1.73.0
rustup default 1.73.0
source "$HOME/.cargo/env"

# Create cargo config to allow unsafe code in tokenizers 0.13.3
echo "Configuring Cargo to allow unsafe code patterns..."
mkdir -p ~/.cargo
cat > ~/.cargo/config.toml << 'EOF'
[build]
rustflags = ["-A", "invalid_reference_casting"]
EOF

echo ""
echo "Current Rust version: $(rustc --version)"
echo "Current Cargo version: $(cargo --version)"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Note: You're currently using Rust 1.73.0 with lint overrides"
echo "To switch back to latest Rust (optional):"
echo "  rustup default stable"
echo "  rm ~/.cargo/config.toml"
echo ""

