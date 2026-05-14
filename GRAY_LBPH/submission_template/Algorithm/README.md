# GRAY-LBPH Submission Template

This folder contains the runtime code expected by the score2026-style benchmark framework.

The public repository intentionally omits generated model artifacts. To build a real submission, train GRAY-LBPH and copy these generated files into this folder:

```text
face_recognizer_model.xml
label_mapping.json
preprocess_config.json
```

Use `preprocess_config.example.json` as the benchmark-oriented GRAY-LBPH preprocessing profile.

Generated private mappings and trained models are local runtime artifacts and must remain outside the source repository.
