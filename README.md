# Science-Quiz-Generation

Welcome to Group's 6 Science Quiz Generation project. This repository contains the code and data for our project.

## Project Structure

In the [notebooks](notebooks) directory, you can find the Jupyter notebooks that either train or evaluate our models. The generated-questions-dataset notebook is used to generate questions using the questions model and adding those to the sciq dataset. The resulting dataset is used for training the options1 model.

In the [scripts](scripts) directory, you can find the scripts that we used train the models on HPC. These scripts do not contain all code that was used for the project, the notebooks do contain all code that was used.

In the [project.ipynb](project.ipynb) notebook, you can find the main notebook that contains the code for the project with the models loaded from [Hugging Face Model Hub](https://huggingface.co/nlp-group-6).


In the [BARTScore](BARTScore) directory, you can find the code we used to calculate the BARTScore.


## Before you start

Before you start, make sure you have `python > 3.10` installed.

To be able to run the notebooks, you need to install the required dependencies. You can do this by running the following command:

```bash
pip install -r requirements.txt
```