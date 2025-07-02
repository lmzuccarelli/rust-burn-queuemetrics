# Overview

Using burn libraries with cuda (gpu) to train and infer a queuemetrics data set 

## Usage

**N.B.**  The data for training, validating and testing is proprietary and has not 
been included in this repo for obvious reasons.

The DataLoader will look for files named

- data/queuemetrics-100000.csv (used for training)
- data/queuemetrics-20000.csv (used for validating)
- data/queuemetrics-1000.csv (used for testing)

Change the directories and file names before compiling.

clone the repo

```
cd rust-burn-metrics

# build 

cargo build --release

#execute

./target/release/rust-burn-metrics train
```
