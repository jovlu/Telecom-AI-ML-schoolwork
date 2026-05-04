# VIM

Small MATLAB project for indoor localization experiments using RSSI data and image-based CNN classification.

## Contents

- `vim1.m` - KNN localization from RSSI measurements.
- `vim2.m` - feed-forward neural network localization.
- `vim3.m` - transfer learning with SqueezeNet on `CNN_dataset/train`.
- `vim4.m` - SVR-based localization with repeated train/test trials.
- `izvestaj*.docx` and `*.pptx` - reports and presentation material.

## Requirements

- MATLAB
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox
- `RSSI Database.xls` for the RSSI scripts
- `CNN_dataset/train` for the CNN script

## Usage

Open the project folder in MATLAB and run the desired script:

```matlab
run("vim1.m")
```

Each script prints evaluation metrics and creates plots for the corresponding experiment.
