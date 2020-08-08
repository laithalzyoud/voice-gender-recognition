# mawdoo3-ai-task

This repository is created to solve the task given by mawdoo3.ai during the interview process.

## Data Specifications

### Resources
After investigating various speech and voice datasets on the web, the following was found to be a suitable dataset:
- Mozilla's Common Voice Dataset: https://www.kaggle.com/mozillaorg/common-voice which is based on the Mozilla Common Vice initiative https://commonvoice.mozilla.org
### Type and Size
The dataset contains valid and invalid data, I'll use only the **valid train** data that was created for training ML models since it doesn't contain noisy data and its size is big enough with around 200,000 utterances. Each voice record is labelled with the corresponding gender, we have 10% of the dataset labelled as females, therefore we will use 10% of the male data to make the data balanced and to avoid overfitting the model to a specific gender. This makes our dataset size around **40,000 utterances with an average of 5 seconds each totalling 55 hours**, which will be enough for the machine learning model to be accurate.

## Data Extraction

- The following dataset was downloaded from Kaggle: https://www.kaggle.com/mozillaorg/common-voice/data?select=cv-valid-train
- The datasets was filtered for utterances that is gender classified
- 
