# SVM and MLP systems for negation cue detection

This repository will contain all code and files used to train a negation detection classifier.
The project is executed by Jona Bosman, Myrthe Buckens, Gabriele Catanese and Eva den Uijl, during January 2021.

### annotations
This folder contains annotations for 10 articles about the vaccination debate that were retrieved from a larger batch of web crawled articles. The annotations were made by the 4 contributing authors on the following files:

* `dc-gov_20170703T010627.txt`
* `cdc-gov_20170706T111717.txt`
* `cid-oxfordjournals-org_20161019T051428.txt`
* `CIDRAP_20161223T092534.txt`
* `Couples-Resorts-Message-Board_20160822T123023.txt`
* `Daily-Intelligencer_20160903T110638.txt`
* `dogsnaturallymagazine-com_20160430T211917.txt`
* `emergency-cdc-gov_20170616T203204.txt`
* `en-wikipedia-org_20170702T222036.txt`
* `fitfortravel-nhs-uk_20160812T165007.txt`

### data
This folder contains the data used for training and testing the system during development. 
For measuring results, the system will be tested on two unseen test sets.

training data: `SEM-2012-SharedTask-CD-SCO-training-simple.txt`

development data: `SEM-2012-SharedTask-CD-SCO-dev-simple.txt`

test data #1: `SEM-2012-SharedTask-CD-SCO-test-cardboard.txt`

test data #2: `SEM-2012-SharedTask-CD-SCO-test-circle-.txt`

### word embeddings
The word embedding model used for the training of the MLP is the "GoogleNewsvectors-negative300.bin.gz".
You can find it here: https://code.google.com/archive/p/word2vec/

### requirements
The packages that are required to run the code for this project can be found in `requirements.txt`.

### code
This folder contains the following scripts and files: 

* `data_statistics.py` prints statistics about the number of tokens and distributions of negations classes of the inputted dataset.

* `preprocessing.py` preprocesses a data file and saves it as a new file with `-preprocessed` at the end.

* `utils.py` contains all functions for the feature extraction.

* `feature_extraction.py` extracts features from a data file and saves it as a new file with `-features` at then end.

* `baseline_system.py` trains a baseline system on a data set with only the token as feature.

* `svm.py` trains a Support Vectors Machine system on the training data with the traditional and morphological features. Saves the best performing system as a new file with `-predictions`at the end.

* `mlp.py` trains a Multilayer Perceptron on the training data with the traditional and morphological features.

* `error-analysis.py` runs an error analysis on the results of the predicted data.

* `stopwords.txt` contains a list of English stopwords.

* `requirements.txt` contains the requirements for the code in this project.


Each of these scripts can be run from the command line through argparse. If you type '-h' after the name of the file, you will get some information regarding the requested arguments.
