# GRAY-LBPH Color Rerank Submission Template

This template is the score2026 runtime for the experimental GRAY-LBPH color-rerank model.

Generated submission assets are intentionally omitted from source control:

- `face_recognizer_model.xml`
- `color_index.npz`
- `label_mapping.json`
- `preprocess_config.json`
- `rerank_runtime_config.json`
- `training_report.json`
- `Algorithm.tar.gz`

The runtime first predicts with GRAY-LBPH, then applies RGB color reranking only inside the configured GRAY top-k candidates.
