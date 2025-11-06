# Warehouse Intelligence System Technical Assessment

This repository contains my solution to the take-home technical assessment.
The data engineering pipeline is contained in the preprocessing/ folder, whilst the subsequent modelling tasks are
each contained in individual directories within the modelling/ folder.
There are supporting configuration, data, and documentation directories.

## Project Overview

This repository includes the data ingestion, configuration, modeling, and documentation components required 
to reproduce the results.

Generic Code Layout

```console
project-root/
│
├── config/                  # Configuration files
│   └── yaml_task_A          # YAML configuration for Task A
|   └── yaml_task_B          # YAML configuration for Task B
│
├── data/                    # Data folder
│   ├── raw_data/            # Raw input data
│   └── processed_data/      # Cleaned or transformed data outputs produced by ingest.py
│
├── docs/                    # Documentation and supporting materials
│   ├── assessment_write_up  # Overall project write-up
│   └── insights_for_task_C  # Analysis or results for Task C
│
├── preprocessing/           # Data ingestion scripts
│   └── ingest.py            # Script for Task A ingestion process
|   └── test/
|       └── test_ingest.py   # Pytest unit test for data ingestation  
│
├── modelling/               # Modelling and analysis code
│   ├── task_A/              # Modelling code for Task A
|   |   └── train.py         # Model training code, with visuals, logs
|   |   └── inference.py     # Example of inference with model
│   └── task_B/              # Modelling code for Task B
|       └── main.py          # Model training code, with visuals, logs
│
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Installation

Clone the repository

```console
git clone warehouse_intelligence_system
cd warehouse_intelligence_system
```


Create and activate a virtual environment (recommended)

```console
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

Install dependencies

```console
pip install -r requirements.txt
```

Should be compatible with any recent python version, i.e. 3.10+

## Running the Code

- Each subtask can be run independently
- The write up is contained in  docs/write_up_slides.pdf. This contains rationale for design decisions
- Parameters can be adjusted in the config/ .yaml files

### Task 1: Data Engineering Pipeline
- **Task**: Build a robust, modular Python script to ingest, clean, and process the 10
days of scan data and the warehouse-layout.json.
- This pipeline is contained in preprocessing/ingest.py
- Associated unit tests are contained in preprocessing/tests/test_ingest.py
- The config for the data reader is config/data_ingest_config.yaml
- To run: python ingest.py or pytest test_ingest.py
- You will be prompted about whether to overwrite existing files if given the same filenames 
- There is info level logging as the script runs
- ingest.py writes files into the data/processed_data/ folder
  - A parquet file containing the combined, "location centric" data set,  timeseries_data.parquet
  - A csv for the warehouse data, warehouse_layout.csv
  - Can optionally produce a full csv of all the combined data, subset_merged_timesteps/csv

### Task 2: Feature Engineering
- **Task**: For each unique Location, build features from its 10-day history.
- The modelling/data_utils ``load_and_preprocess function" takes the parquet file and produces features for use in training
  - Categorical
  - Numerical
  - Some features are dropped

### Task 3: Error Prediction Model
- **Task**: Build a model to predict the likelihood of an error (Status != Correct) for a
given Location on the next day
- The timeseries_data.parquet file is used to train this model
- Code for this task is found in modelling/error_prediction, including
  - The model itself
  - Dataset definition 
  - Other functions for data processing, test/validation/train and evaluation
- The config for the model, including hyperparameters, is config/error_prediction_config.yaml
- A pretrained model and checkpoint is already contained within the folder, referenced by best_model.pth
- To retrain, delete this checkpoint
- To train, python train.py. This will print some statistics and plot a confusion matrix
- As the model trains, the performance is logged at info level
- For an example of inference with a trained model (best_model.pth), run python inference.py
- An output test confusion matrix is kept in the docs folder


### Task 4 i: Spatial Error Clustering
- **Task**: Use the (x, y, z) coordinates of historical errors to
automatically identify &quot;problem zones&quot; that may span multiple locations.
- Code for this task is found in modelling/spatial_error_clustering, including
  - The clustering algorithm
  - Visuals
  - Utility functions
- The config for the model, including hyperparameters, is config/clustering_config.yaml
- To run, python main.py, this will produce a warehouse map showing the clusters
- An output warehouse map is kept in the docs folder

### Task 4 ii: Item Co-location &amp; Pick-Path Analysis
- **Task**: Analyse the Barcode data. Use
association graph analysis, or another method to find items that are frequently
stored together, picked together (inferred), or misplaced near each other
- Code for this task is found in modelling/barcode_analysis
- To run, python main.py
- This will produce an associated graph of misplaced items, and produce some statistics for different relationships
- An example of this graph is kept in the docs folder, and insights are found in docs/barcode_analysis_insights.md

### Task 4 iii: Time Series Anomaly Detection
- **Task**: Item Co-location &amp; Pick-Path Analysis
- Code for this task is found in modelling/barcode_analysis
- To run the model, python model.py. This will produce a .npz file containing the model parameters
- To run inference, python inference.py. This will take the produced .npz file, load the model and 
provide an inference example
- Running main.py produces a graph of detected anomalies, an example is stored in docs/ folder


## Author

Samuel Bennett
GitHub: samuelbennett1020

Email: samuelbennett1020@gmail.com

## Notes

All data files used in this project are stored locally under data/.

Configuration parameters, stored as yaml files in config/ can be customized without modifying code directly.

Each subtask folder is self-contained and can be executed independently for testing or evaluation, but there must
be a timeseries_data.parquet file.
