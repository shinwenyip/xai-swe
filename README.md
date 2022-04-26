# Comparison of XAI Tooling for Software Defect Prediction

This repository contains the code for conducting experiments to perform empirical comparison between SHAP, LIME and PyExplainer. 

## Installation
1. **Download/Clone the code repository**  (https://anonymous.4open.science/r/xai-swe-235E)
2. **Install and activate conda environment.**

   Firstly, install Anaconda from [here](https://docs.anaconda.com/anaconda/install/) if it is
not already installed on your machine.

   In your terminal, verify that conda is installed by running:

   ```bash
     conda --version
   ```
   To create a working environment for running the project code:
   (make sure you are in the xai-swe directory where the env.yml file is located)
   ```bash
     conda activate xaienv
     conda env create -f env.yml
   ```

3. **View Experiment results**

   Open `charts.ipynb` in Anaconda environment and run code cells for each metric to view
the results generated for each metric comparison


## Usage

Below is an outline of the files within the repository and their respective functionalities.

**functions.py**
- This python file is where all the XAI evaluation metrics (Section 3.4) are implemented as
individual functions, along with some helper functions used to generate explanations for our
experiment

**project.py**
- This python file is where the Project class is defined in order to encapsulate the data (name,
models, test set, train set) and methods that can be used for each project dataset. get datasets
is used to return training and testing data from ”datasets” directory depending on the name of
requested project (example: ANT, CAMEL etc.). set train test uses get datasets to set
up the data for the current project. train global model trains and sets a ML model using
training data for a particular project.get sampled data takes in the name of global ML task
model as input and returns 100 balanced sample instances from the test set of the project.

**examples.ipynb**
- This python jupyter notebook file is used to show how to visualise examples of explanation
output for LIME, SHAP and PyExplainer.

**experiment.ipynb**
- This python jupyter notebook file is where explanations used in the experiments are generated
(from the 100 samples for each ML model in each project) and saved into files in the explanations
directory (explanations directory have to created beforehand)

**eval.ipynb**
- This python jupyter notebook file is used to run the XAI evaluation functions defined in
functions.py on the generated explanations saved in the explanations directory. (have to create
explanations directory and run experiment.ipynb first)

**charts.ipynb**
- This python notebook is used to display results for each metric as charts using the results data
saved in the eval results directory.

### Adding Different Datasets
---
To add more datasets, first add the csv files for test data and train data (can also be
split later) into the ”datasets” directory. Then, within the project.py file get datasets
method, define a name for that dataset project and set ds train and ds test to the respective filenames. For example:
```python
elif self.name == "POI":
    ds_train,ds_test = "poi-2.5.csv","poi-3.0.csv"
```
The particular project can then be initialised in any file (where project module is imported)
by running:
```python
poi_project = Project("POI")
poi_project.set_train_test()
```

### Adding different ML models
---
To add other types of ML model (such as Decision Trees, Neural Networks etc) for experimentation, a conditional statement can be added to the train global model method in
project.py, with a string associated with the model. For example:
```python
elif model name == "DT":
    global_model = DecisionTreeClassifier(random_state=self.random_state)
    global_model.fit(self.X train.values,self.y_train.values)
```
The trained model will then be added to list of models associated with the project
when the method is called and the model can be accessed by running:
```python
poi_project.train_global_model("DT")
global_model = poi_project.models["DT"]
```
