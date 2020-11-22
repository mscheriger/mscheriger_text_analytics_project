# Text Analytics Project - Question Answering using BERT 
### Michael Scheriger

## Overview
This project attempts to use Information Retrieval and BERT to answer multiple choice questions. To run the application, please follow the instructions below.

## Reproducing results

### Download the data
The data is located in https://allenai.org/data/arc. Download the data and place the unzipped folder ARC-V1-Feb2018-2 in the data directory. 

### Configurations
In config.yaml, there are four options that you can configure. 
- "use_rouge" determines if cosine or rouge score is used in the BERT method
- "train" determines if the training or test data is used
- "ten" determines if 5 or 10 lines are used for context in the Bert method
- "lines" are the numeric value of lines to return for context. If ten is true, lines should be 10

### Build Elasticsearch
Both the information retrieval and BERT methods utilize Elasticsearch, so you will need Elasticsearch downloaded and running in order to build the database. Once running, run the below command that will run Elasticsearch.

```bash
python run.py -b
```

### Get context (Optional)
The BERT method requires context for each question, which is found using Elasticsearch. However, there are files in the obj folder that already have the context, so running this is optional. To get the context, run the following command.

```bash
python run.py -g
```

### Run IR Method (Optional)
The obj folder has files that contain the results of the IR method. To reproduce the results, run the following command.

```bash
python run.py -i
```

### Run BERT Method (Optional)
The obj folder has files that contain the results of the BERT method. To reproduce the results, run the following command.

```bash
python run.py -b
```

### See results
To see the accuracy of each model on the hard and easy questions for training and test data, run the following command.

```bash
python run.py -p
```

## Run the API
In order for the API to run, be sure to have your Elasticsearch database built with the ARC corpus. Otherwise, the API will return an error. To run the api, run the following command:

```bash
python api.py
```

Then visit "http://127.0.0.1:5000/apidocs" (this may just be for windows - whatever your localhost is go to port 5000). To try out the model, enter multiple choice options in the A, B, C, and D fields, the question in the question field, and which method to use. For methods, choose either "IR" or "BERT", I recommend using the IR method.
