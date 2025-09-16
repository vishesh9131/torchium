#!/bin/bash

# Deploy Torchium Documentation to GitHub Pages
# This script builds the documentation and deploys it to the gh-pages branch

set -e

echo "🚀 Building and deploying Torchium documentation to GitHub Pages..."

# Build the documentation
echo "📚 Building documentation..."
cd torchium/docs
sphinx-build -b html source build/html

# Create a temporary directory for deployment
TEMP_DIR=$(mktemp -d)
echo "📁 Using temporary directory: $TEMP_DIR"

# Clone the repository in the temp directory
cd "$TEMP_DIR"
git clone https://github.com/vishesh9131/torchium.git
cd torchium

# Checkout or create gh-pages branch
if git show-ref --verify --quiet refs/heads/gh-pages; then
    git checkout gh-pages
    git pull origin gh-pages
else
    git checkout --orphan gh-pages
    git rm -rf .
fi

# Copy the built documentation
cp -r /Users/visheshyadav/Documents/GitHub/torchium/torchium/docs/build/html/* .

# Add and commit changes
git add .
git commit -m "Deploy documentation $(date)"

# Push to GitHub Pages
echo "🌐 Deploying to GitHub Pages..."
git push origin gh-pages

# Cleanup
cd /Users/visheshyadav/Documents/GitHub/torchium
rm -rf "$TEMP_DIR"

echo "✅ Documentation deployed successfully!"
echo "📖 Your documentation will be available at: https://vishesh9131.github.io/torchium/"
echo ""
echo "Note: It may take a few minutes for GitHub Pages to update."
