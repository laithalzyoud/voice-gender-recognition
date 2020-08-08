# mawdoo3-ai-task

## Summary
This repository is created to solve the task given by mawdoo3.ai during their interview process. It contains necessary code to pre-process data, extract audio features based on the mel-spectogram and train models to classify gender based on audio features. All the metnioned processes was exposed with RESTFul API's using Python web framework Flask. A Docker image is also available to run this project out of the box :zap:

## Data Specifications

### Resources
After investigating various speech and voice datasets on the web, the following was found to be a suitable dataset:
- Mozilla's Common Voice Dataset: https://www.kaggle.com/mozillaorg/common-voice which is based on the Mozilla Common Vice initiative https://commonvoice.mozilla.org
### Type and Size
The dataset contains valid and invalid data, I'll use only the **valid train** data that was created for training ML models since it doesn't contain noisy data and its size is big enough with around 200,000 utterances. Each voice record is labelled with the corresponding gender, we have 10% of the dataset labelled as females, therefore we will use 10% of the male data to make the data balanced and to avoid overfitting the model to a specific gender. This makes our dataset size around **40,000 utterances with an average of 5 seconds each totalling 55 hours**, which will be enough for the machine learning model to be accurate.

## Data Extraction

- The following dataset was downloaded from Kaggle: https://www.kaggle.com/mozillaorg/common-voice/data?select=cv-valid-train, you need to update the [config](https://github.com/laithalzyoud/mawdoo3-ai-task/blob/master/config.json) with the new path for the downloaded dataset.
- The dataset was filtered for utterances that is only gender classified as male or female
- The dataset was balanced based on the minimum # of gender occurrences to avoid bias in the trained models

API: `/filter`
