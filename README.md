# Overview

Using burn libraries with cuda (gpu) to train and infer a queuemetrics data set 

## Usage

**N.B.**  The data for training, validating and testing you will need to copy data into the data folder.

Use the following repo to create "synthetic data" 

- https://github.com/lmzuccarelli/rust-syntheticdata-code-generator 

The DataLoader will look for files named (change and recompile needed)

- data/queuemetrics-100000.csv (used for training)
- data/queuemetrics-20000.csv (used for validating)
- data/queuemetrics-1000.csv (used for testing)

Change the directories and file names before compiling.

clone the repo

```
cd rust-burn-queuemetrics

# build 

cargo build --release

#execute (train)

./target/release/rust-burn-queuemetrics --config app-config.json train

# execute (inference)

./target/release/rust-burn-queuemetrics --config app-config.json inference

# execute (serve)

./target/release/rust-burn-queuemetrics --config app-config.json serve

 # create a simple json file to check the prediction
cat <<EOF > queuemetrics.json 
{
  "processors": 4,
  "avg_batch_size": 416.95,
  "queue_length": 2.33,
  "processing_time": 2.39,
  "status": 0
}

curl -k -d'&queuemetrics.json' https://localhost:8085/inference
```

## Certs

A script is included to create the key pairs for this service (tls)

It will copy a rootCA.pem file and update the local CA trust

**N.B.** Update the script (i.e -subj section to your requirments). The script is "fedora" specific, change it for your distro (espescially the trusted CA location) 

Execute it with the following parameters

```
scripts/create-key-pair.sh <hostname> <ip>
```
