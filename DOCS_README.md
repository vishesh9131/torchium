# Torchium Documentation

This directory contains the complete documentation for the Torchium library, built using Sphinx.

## 📚 Documentation Structure

```
torchium/docs/
├── source/                 # Source files for documentation
│   ├── api/               # API reference documentation
│   ├── examples/          # Usage examples and tutorials
│   ├── tutorials/         # Step-by-step tutorials
│   ├── conf.py           # Sphinx configuration
│   └── index.rst         # Main documentation index
├── build/                 # Built documentation (generated)
│   └── html/             # HTML output
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Build Documentation Locally

```bash
# Install dependencies
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints

# Build HTML documentation
cd torchium/docs
sphinx-build -b html source build/html
```

### 2. Serve Documentation Locally

#### Option A: Using the provided script
```bash
python serve_docs.py [port]
# Default port is 8000
```

#### Option B: Using Python's built-in server
```bash
cd torchium/docs/build/html
python -m http.server 8000
```

Then open http://localhost:8000 in your browser.

## 🌐 Hosting Options

### Option 1: GitHub Pages (Recommended)

1. **Automatic Deployment**: The repository includes a GitHub Actions workflow that automatically builds and deploys documentation when you push to the main branch.

2. **Manual Deployment**: Use the provided script:
   ```bash
   ./deploy_docs.sh
   ```

3. **Access**: Your documentation will be available at:
   - https://vishesh9131.github.io/torchium/

### Option 2: Read the Docs

1. Connect your GitHub repository to Read the Docs
2. The documentation will be automatically built and hosted
3. Access at: https://torchium.readthedocs.io

### Option 3: Netlify/Vercel

1. Connect your repository to Netlify or Vercel
2. Set build command: `cd torchium/docs && sphinx-build -b html source build/html`
3. Set publish directory: `torchium/docs/build/html`

## 📝 Documentation Content

The documentation includes:

- **Quick Start Guide**: Getting started with Torchium
- **API Reference**: Complete API documentation for all optimizers and loss functions
- **Tutorials**: Step-by-step guides for different use cases
- **Examples**: Real-world usage examples
- **Performance Guide**: Optimization tips and benchmarks

## 🔧 Customization

### Adding New Documentation

1. **New Tutorial**: Add a `.rst` file in `source/tutorials/`
2. **New Example**: Add a `.rst` file in `source/examples/`
3. **Update Index**: Add references to new files in `source/index.rst`

### Modifying Configuration

Edit `source/conf.py` to:
- Change theme settings
- Add new extensions
- Modify build options
- Update project information

### Styling

The documentation uses the Read the Docs theme. You can customize it by:
- Adding custom CSS in `source/_static/`
- Modifying theme options in `conf.py`

## 🛠️ Development

### Rebuilding Documentation

After making changes to the source files:

```bash
cd torchium/docs
sphinx-build -b html source build/html
```

### Cleaning Build Files

```bash
cd torchium/docs
rm -rf build/
```

### Checking for Errors

```bash
cd torchium/docs
sphinx-build -b html source build/html -W
```

The `-W` flag treats warnings as errors.

## 📋 Available Commands

- `python serve_docs.py` - Serve documentation locally
- `./deploy_docs.sh` - Deploy to GitHub Pages
- `sphinx-build -b html source build/html` - Build HTML documentation
- `sphinx-build -b latex source build/latex` - Build PDF documentation

## 🎯 Features

- **Responsive Design**: Works on desktop and mobile
- **Search Functionality**: Built-in search across all documentation
- **Code Highlighting**: Syntax highlighting for Python code
- **Cross-References**: Automatic linking between sections
- **API Documentation**: Auto-generated from docstrings
- **Multiple Formats**: HTML, PDF, and other formats supported

## 📞 Support

If you encounter issues with the documentation:

1. Check the build logs for errors
2. Ensure all dependencies are installed
3. Verify the source files are valid RST format
4. Open an issue on GitHub

## 🔄 Updating Documentation

To update the documentation:

1. Make changes to source files in `source/`
2. Rebuild: `sphinx-build -b html source build/html`
3. Test locally: `python serve_docs.py`
4. Deploy: `./deploy_docs.sh` (for GitHub Pages)

The documentation is automatically updated when you push changes to the main branch (if using GitHub Actions).
