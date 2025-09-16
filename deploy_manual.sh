#!/bin/bash

# Manual deployment script for GitHub Pages
# Run this if GitHub Actions fails

echo "🚀 Manual deployment to GitHub Pages..."

# Build documentation
echo "📚 Building documentation..."
cd torchium/docs
sphinx-build -b html source build/html
cd ../..

# Create deployment branch
echo "🌿 Setting up deployment..."
git checkout -B gh-pages

# Remove everything except docs
find . -maxdepth 1 ! -name '.git' ! -name 'torchium' ! -name '.' -exec rm -rf {} \;

# Copy built docs to root
cp -r torchium/docs/build/html/* .

# Add .nojekyll to prevent Jekyll processing
touch .nojekyll

# Commit and push
echo "📤 Deploying..."
git add .
git commit -m "Deploy documentation - $(date)"
git push -f origin gh-pages

# Return to main branch
git checkout main

echo "✅ Deployment complete!"
echo "🌐 Documentation will be available at: https://vishesh9131.github.io/torchium/"
