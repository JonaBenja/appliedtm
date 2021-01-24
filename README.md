# A [bunch of systems] for negation cue detection

This repository will contain all code and files used to train a negation detection classifier.
The project is executed by Jona Bosman, Myrthe Buckens, Gabriele Catanese and Eva den Uijl, during January 2021.

### Annotations
10 articles about vaccination that were retrieved from a larger batch of web crawled articles.

`annotations` folder: contains the annotations made by the 4 contributing authors on the following files:

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

### Data
This folder contains the data used for training and testing the system during development. 
For measuring results, the system will be tested on two unseen test sets.

training data: `SEM-2012-SharedTask-CD-SCO-training-simple.txt`

development data: `SEM-2012-SharedTask-CD-SCO-dev-simple.txt`

test data: `SEM-2012-SharedTask-CD-SCO-test-cardboard.txt`

&

`SEM-2012-SharedTask-CD-SCO-test-circle-.txt`

### Code
This folder contains the following scripts:

* `preprocessing.py` preprocesses the data and saves it as new files.

training data: `SEM-2012-SharedTask-CD-SCO-training-simple-preprocessed.conll`

development data: `SEM-2012-SharedTask-CD-SCO-dev-simple-preprocessed.conll`

test data: `SEM-2012-SharedTask-CD-SCO-test-cardboard-preprocessed.conll`

&

`SEM-2012-SharedTask-CD-SCO-test-circle-preprocessed.conll`

* `utils.py` contains all functions for the feature extraction.

* `feature_extraction.py` extracts features and writes them to new files:

training data: `SEM-2012-SharedTask-CD-SCO-training-preprocessed-features.conll`

development data: `SEM-2012-SharedTask-CD-SCO-dev-preprocessed-features.conll`

test data: `SEM-2012-SharedTask-CD-SCO-test-cardboard-preprocessed-features.conll`

&

`SEM-2012-SharedTask-CD-SCO-test-circle-preprocessed-features.conll`

* `baseline_system.py`

* `svm_classifier.py`

* `mlp_classifier.py`

* `stopwords.txt`
