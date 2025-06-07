# sibyllai-core

Audio-spotting & mood-analysis engine for Motive-AI.

## Dependencies and Environment Considerations

This project relies on several large machine learning and audio processing libraries.
Dependencies such as PyTorch, TensorFlow, Essentia, and their associated components can consume a significant amount of disk space (potentially several gigabytes).

Users are advised to:
- Ensure adequate disk space is available before starting the installation.
- Maintain a stable internet connection during the installation process due to the size of the downloads.
- Check the `pyproject.toml` file for specific Python version requirements (currently Python >=3.11) and a detailed list of all dependencies.

Setting up a virtual environment (e.g., using `venv` or `conda`) is highly recommended to manage dependencies and avoid conflicts with other Python projects.
