# Data

Expected inputs:

- `RSSI Database.xls` - RSSI measurements and known `(x, y)` coordinates.
- `CNN_dataset/` - extracted image dataset for the CNN experiment.

The CNN folder layout should be:

```text
data/CNN_dataset/
  train/<class-name>/<image-files>
  valid/<class-name>/<image-files>
  test/<class-name>/<image-files>
```

The extracted image dataset is ignored by Git. Keep only a small sample or a compressed archive in the repository if you need reviewers to run the CNN experiment directly.
