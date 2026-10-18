# Notebooks

Use notebooks only for reproducible exploration. Move reusable logic into tested Python modules, clear sensitive outputs before commit, and never store raw datasets, checkpoints, credentials, or large generated artifacts here.

`classical_ml_colab.ipynb` is the supported Google Colab entry point. The
reusable training and audit logic remains in `backend/` and `scripts/`.
