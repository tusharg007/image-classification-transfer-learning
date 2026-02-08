# Contributing to Image Classification Transfer Learning

Thank you for your interest in contributing to this project! This document provides guidelines for contributions.

## Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/image-classification-transfer-learning.git
   cd image-classification-transfer-learning
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Install Development Tools**
   ```bash
   pip install black flake8 isort pytest pytest-cov
   ```

## Development Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow these coding standards:

**Python Style:**
- Follow PEP 8
- Use Black for formatting: `black .`
- Use isort for imports: `isort .`
- Maximum line length: 100 characters

**Documentation:**
- Add docstrings to all functions/classes
- Use Google-style docstrings
- Update README if adding features

**Type Hints:**
```python
def process_image(
    image_path: str,
    target_size: Tuple[int, int]
) -> np.ndarray:
    """
    Process image for model input.
    
    Args:
        image_path: Path to image file
        target_size: Target dimensions (height, width)
        
    Returns:
        Preprocessed image array
    """
    pass
```

### 3. Test Your Changes

Run tests before committing:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_data_loader.py
```

### 4. Code Quality Checks

```bash
# Format code
black src/ scripts/ tests/

# Sort imports
isort src/ scripts/ tests/

# Check style
flake8 src/ scripts/ tests/

# Type checking (optional)
mypy src/
```

### 5. Commit Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: description of feature"
```

**Commit Message Format:**
```
<type>: <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```
feat: Add EfficientNet model support

- Implemented EfficientNet B0-B7 variants
- Added configuration files
- Updated documentation

Closes #123
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Types of Contributions

### 🐛 Bug Reports

When reporting bugs, include:
- Python and TensorFlow versions
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces
- Minimal code example

**Template:**
```markdown
**Describe the bug**
A clear description of the bug.

**To Reproduce**
Steps to reproduce:
1. Load data using...
2. Train model with...
3. Error occurs...

**Expected behavior**
What you expected to happen.

**Environment**
- OS: Ubuntu 20.04
- Python: 3.8.10
- TensorFlow: 2.8.0
- GPU: NVIDIA RTX 3080

**Additional context**
Any other relevant information.
```

### ✨ Feature Requests

When suggesting features:
- Clear use case description
- Expected behavior
- Alternative solutions considered
- Willingness to implement

### 🔧 Code Contributions

**Good First Issues:**
- Adding new augmentation techniques
- Implementing additional metrics
- Improving documentation
- Adding unit tests
- Bug fixes

**Advanced Contributions:**
- New model architectures
- Performance optimizations
- Distributed training support
- Model compression techniques

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] CHANGELOG updated (if applicable)
- [ ] No merge conflicts

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How was this tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added to complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
```

### Review Process

1. Automated checks run (tests, linting)
2. Code review by maintainers
3. Address feedback if requested
4. Approval and merge

## Testing Guidelines

### Writing Tests

```python
import pytest
import numpy as np
from src.data.data_loader import ImageDataLoader

def test_data_loader_initialization():
    """Test DataLoader initialization."""
    loader = ImageDataLoader(
        image_size=(224, 224),
        batch_size=32
    )
    assert loader.image_size == (224, 224)
    assert loader.batch_size == 32

def test_image_preprocessing():
    """Test image preprocessing."""
    loader = ImageDataLoader()
    # Create dummy image
    dummy_image = np.random.rand(224, 224, 3) * 255
    processed = loader.preprocess_image(dummy_image)
    
    # Check normalization
    assert processed.min() >= 0
    assert processed.max() <= 1
```

### Test Coverage

Aim for:
- Unit tests: >80% coverage
- Integration tests for major workflows
- Edge cases and error handling

## Documentation

### Code Documentation

```python
def train_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 100
) -> keras.callbacks.History:
    """
    Train image classification model.
    
    This function implements the complete training pipeline with
    callbacks for early stopping and model checkpointing.
    
    Args:
        model: Keras model to train
        train_ds: Training dataset with batched images and labels
        val_ds: Validation dataset for monitoring performance
        epochs: Maximum number of training epochs (default: 100)
        
    Returns:
        History object containing training metrics
        
    Raises:
        ValueError: If datasets are empty or incompatible
        
    Example:
        >>> model = get_model('resnet50')
        >>> history = train_model(model, train_ds, val_ds, epochs=50)
        >>> print(f"Best accuracy: {max(history.history['val_accuracy'])}")
    """
    pass
```

### README Updates

When adding features, update:
- Installation instructions
- Usage examples
- API documentation
- Configuration options

## Release Process

Maintainers follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR:** Breaking changes
- **MINOR:** New features (backward compatible)
- **PATCH:** Bug fixes

## Questions?

- Open an issue for discussion
- Check existing issues/PRs
- Contact maintainers

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

Thank you for contributing! 🎉
