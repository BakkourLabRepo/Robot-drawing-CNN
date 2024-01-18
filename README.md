# Robot drawing CNN
This project aims to develop and train a convolutional neural network (CNN) model to detect features in the robots drawn. 


## Files Description
- `Helper_Functions.py`: Contains utility functions used for 
`Run_CNN_Model.ipynb` and `Run_CNN_Model.py`. This file includes data formatting, preprocessing, and augmentation. It also includes spliting for training, validating, and testing data, followed by mertircs to evaluate model performance (as well as visualization). 

- `Run_CNN_Model.py`: The Python script version of the Jupyter notebook for training and evaluating the CNN model. Suitable for running in Midway3. 

- `Run_CNN_Model.ipynb`: A Jupyter notebook illustrating the workflow of training and evaluating the Convolutional Neural Network (CNN) model. Includes code with annotations and visualizations.
  
- `Run_CNN.sbatch`: A sbatch file for running on midway.

- `Model Performance.txt`: A text file summarizing the performance metrics of the CNN model. Includes data such as loss, accuracy, precision, and recall on validation and test sets.

- `combs`: Directory containing a sample set of images used for code testing on the CPU environment. 

- `confusion_matrices.jpg`: An image file showing the confusion matrices of the model, which helps in understanding the model's performance in terms of false positives and false negatives.

- `testing_data_prediction.csv`: Contains the predictions made by the model on the test dataset alongside the true labels for evaluation.

- `__pycache__`: A directory that Python uses to store precompiled bytecode, speeding up subsequent program startup.


## Usage
This project can be run using either the Python script or the Jupyter Notebook (on a CPU environemnt).

### Running the Python Script

To run the Python script, use the following command in your terminal:

```bash
python Run_CNN_Model.py
```

### Running the Jupyter Notebook
To view and run the Jupyter Notebook, first make sure you have Jupyter installed. If not, you can install it using pip:

```bash
pip install notebook
```

Then, launch Jupyter Notebook:
```bash
jupyter notebook
```

In the Jupyter interface, navigate to Run_CNN_Model.ipynb and open it. You can run the cells in the notebook to train and evaluate the model interactively.


## Things to work with
1. Handling of positive weights (class imbalance)
2. The classification matrix is not in the anticipated structure
3. Change the (relative) path of stimulus in python script
4. Handle parallel processing and gpu issues (for tensorflow)
5. Find out the modules that needs to be load and unload in sbatch script
6. Run the sample dataset on midway
7. Use VGG model on midway
8. Issues with predicted_labels_string (all "bo_1_bo_6")



