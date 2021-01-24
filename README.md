# An SVM and MLP system for negation cue detection

This repository will contain all code and files used to train a negation detection classifier.
The project is executed by Jona Bosman, Myrthe Buckens, Gabriele Catanese and Eva den Uijl, during January 2021.

### annotations
This folder contains annotations for 10 articles about vaccination that were retrieved from a larger batch of web crawled articles. The annotations were made by the 4 contributing authors on the following files:

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

### code
This folder contains the following scripts:

* `data_statistics.py` prints statistics about the number of tokens and distributions of negations classes of the inputted dataset.

* `preprocessing.py` preprocesses a data file and saves it as a new file with `-preprocessed` at the end.

* `utils.py` contains all functions for the feature extraction.

* `feature_extraction.py` extracts features from a data file and saves it as a new file with `-features` at then end.

* `baseline_system.py` trains a baseline system on a data set with only the token as feature.

* `svm_classifier.py` trains a Support Vectors Machine system on the training data.

* `mlp_classifier.py` trains a Multilayer Perceptron on the training data.

* `stopwords.txt` contains a list of English stopwords.
