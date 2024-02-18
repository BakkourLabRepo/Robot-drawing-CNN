# Robot drawing CNN
This project aims to develop and train a convolutional neural network (CNN) model to detect features in the robots drawn. 


## Files Description
- `Helper_Functions.py`: Contains utility functions used for 
`Run_CNN_Model.ipynb` and `Run_CNN_Model.py`. This file includes data formatting, preprocessing, and augmentation. It also includes spliting for training, validating, and testing data, followed by mertircs to evaluate model performance (as well as visualization). 

- `Run_CNN_Model.py`: The Python script version of the Jupyter notebook for training and evaluating the CNN model. Suitable for running in Midway3. 

- `Run_CNN_cpu_ssd`: A sbatch file for running on midway (on CPU).

- `Run_CNN_gpu_ssd`: A sbatch file for running on midway (on GPU).

- `combs`: Directory containing a sample set of images used for code testing on the CPU environment. 

- `confusion_matrices.jpg`: An image file showing the confusion matrices of the model, which helps in understanding the model's performance in terms of false positives and false negatives.


## Usage
This project can be run using either the Python script or the Jupyter Notebook (on a CPU environemnt).

### Running the Python Script

To run the Python script, use the following command in your terminal:

```bash
python Run_CNN_Model.py
```


## Things to work with
1. Handling of positive weights (class imbalance)
2. The classification matrix is not in the anticipated structure
3. Handle parallel processing and gpu issues (for tensorflow)
4. Find out the modules that needs to be load and unload in sbatch script
5. Run the sample dataset on midway
5. Issues with predicted_labels_string (all "bo_1_bo_6")



