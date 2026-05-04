# Indoor Localization Experiments

This repository contains MATLAB experiments for indoor localization using Wi-Fi RSSI measurements, plus a small CNN transfer-learning experiment.

The main task is to estimate a physical `(x, y)` position from signal-strength readings collected from multiple access points. The project compares several approaches:

- K-nearest neighbors on normalized RSSI vectors
- Feed-forward neural network regression
- Support vector regression for `x` and `y` coordinates
- CNN transfer learning with SqueezeNet for image-based classification

## Repository Structure

```text
src/
  knn_localization.m
  neural_network_localization.m
  svr_localization.m
  cnn_transfer_learning.m

data/
  RSSI Database.xls
  README.md

docs/
  assignments/
  reports/
  presentations/
  assets/

results/
  README.md
```

## Data

The RSSI scripts expect:

```text
data/RSSI Database.xls
```

The first columns contain RSSI readings from access points. Missing access points are stored as `*` and converted to `-120 dBm`. The final two columns contain the measured location coordinates.

The CNN script expects the image dataset in this layout:

```text
data/CNN_dataset/
  train/<class-name>/*.jpg
  valid/<class-name>/*.jpg
  test/<class-name>/*.jpg
```

The extracted CNN dataset is ignored by Git because it contains many generated image files. If sharing this repository publicly, either provide a small sample dataset, include a download link, or add back a compressed archive such as `data/CNN_dataset.zip`.

## Requirements

- MATLAB
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox
- Deep Learning Toolbox Model for SqueezeNet Network

## Running

Open the repository root in MATLAB and run one of the scripts:

```matlab
run("src/knn_localization.m")
run("src/neural_network_localization.m")
run("src/svr_localization.m")
run("src/cnn_transfer_learning.m")
```

Each script prints evaluation metrics and creates plots for the experiment.

## Notes

The code is intentionally kept as simple MATLAB scripts because the focus is on comparing models and preprocessing choices, not building a production application. Generated plots and exported metrics should be placed in `results/`.
