import tensorflow as tf
# import tensorflow_addons as tfa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, \
    multilabel_confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn
import math
import os
import re
from PIL import Image


#################### Helper functions for the robot task #######################
def images_and_labels(folder_path, feature_num, class_num, labels):
    """
    Outputs 1) a numpy array where the first dimension equals to the number of 
    robots and the second and third dimensions equal to the pixel*pixel size of 
    the robot images; and 2) a pandas dataframe where number of rows equals to
    number of robots and number of columns equals to total number of 
    "feature * class" combinations.

    Inputs:
        folder_path(str): directory where the robot images are stored 
        feature_num (int): number of features
        class_num (int): number of options per class
        labels (lst): all possible labels ("feature * class_num" combinations)
        
    Returns: 1) an numpy array of image pixel intensities; and
        2) a pandas dataframe consisting of rows of "feature vectors" of robots
    """
    
    # Create an empty list to store individual image arrays
    image_lst = []

    # Initialize column names for the dataFrame
    label_df = pd.DataFrame(columns=labels)

    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            if f.endswith('.png'):
                full_path = os.path.join(dirpath, f)
                # Load image files into a 2D numpy array where 
                # each value represents a pixel' intensity (from 0 to 255)
                image_pixels = Image.open(full_path)
                image_lst.append(np.asarray(image_pixels))

                # Initilize a row of observation in dataframe as a zero vector
                next_row = len(label_df)
                label_df.loc[next_row] = [0] * (feature_num * class_num)
                # A pattern to extract from the png file name (E.g., "bo_1")
                matches = re.findall(r'([a-z]+)-(\d+)', f)
                for feature, num in matches:
                    if int(num) != 0: 
                        label_df.at[next_row, f"{feature}_{num}"] = 1
    
    # Convert the list of image arrays to a single numpy array
    image_array = np.stack(image_lst, axis=0)

    return image_array, label_df


def data_split(data, split_ratio, random_state):
    """
    Split dataset into training, validation, and testing based on split ratio.

    Inputs:
        data (numpy array or pandas dataframe): data to be splited
        split_ratio (tuple): split ratio training, validation, testing dataset
        random_state (int): for reproducibility of data spliting
    
    Returns (same data type as `data`): data_train,data_val, data_test
    """
    
    train, val, test = split_ratio
    data_train, data_temp, = train_test_split(data, test_size=1 - train,
                                              random_state = random_state)
    data_val, data_test = train_test_split(data_temp, 
                                           test_size=test / (val + test),
                                           random_state = random_state)

    return data_train, data_val, data_test


def preprocessing(image_array, label_df, 
                  split_ratio, random_state, 
                  batch_size):
    """
    Preprocesses the image_array (normalized to [0,1]) and label_df into 
    tensorflow objects that are splited into training, validation, and 
    testing data objects. Moreover, creates data generator (applies batching,
    shuffling, and prefetching transformations) for model traininig. 

    Inputs:
        image_array (numpy array): image pixel intensities
        label_df (pandas dataframe): rows of "feature vectors" of robots
        split_ratio (tuple): split ratio training, validation, testing dataset
        random_state (int): for reproducibility of data spliting
        batch_size (int): batch size

    Returns (tensorflow): training dataset, validation dataset, testing dataset
    """
    
    # Normalize pixel intensities to [0,1]
    image_array = image_array / 255.0

    # Split dataset according to split ratio
    image_train, image_val, image_test = data_split(image_array, split_ratio,
                                                    random_state = random_state)
    label_train, label_val, label_test = data_split(label_df, split_ratio,
                                                    random_state = random_state)
    
    # Convert to TensorFlow datasets (including both images and labels)
    train_data = tf.data.Dataset.from_tensor_slices((image_train, label_train))
    val_data = tf.data.Dataset.from_tensor_slices((image_val, label_val))
    test_data = tf.data.Dataset.from_tensor_slices((image_test, label_test))

    # Apply data augmentation (only) to the training dataset
    # train_data = train_data.map(data_augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batching, shuffling, and prefetching
    prefetch_size = tf.data.experimental.AUTOTUNE
    train_data = (train_data
                  .shuffle(buffer_size=len(image_train))
                  .batch(batch_size)
                  .prefetch(buffer_size=prefetch_size))
    val_data = (val_data
                .batch(batch_size)
                .prefetch(buffer_size=prefetch_size))
    test_data = (test_data
                .batch(batch_size)
                .prefetch(buffer_size=prefetch_size))

    return train_data, val_data, test_data


# def data_augmentation(image, label, noise_mean=0.0, noise_std=0.1):
#     """
#     Performsn several data augmentation techniques to the image. 

#     Inputs:
#         1) train_data (numpy array): image
#         2) label (numpy array): feature vector associated with the image
#         3) noise_mean (float): mean of Gaussian random noise
#         4) noise_std (float): standard deviation of Gaussian random noise
    
#     Returns (numpy array): image with data augmentation
#     """

#     # Reshape the black-and-white image to add a channel dimension
#     image = tf.expand_dims(image, -1)

#     # Random flips
#     image = tf.image.random_flip_left_right(image)
#     image = tf.image.random_flip_up_down(image)

#     # Random rotations (random angles)
#     rotate_angle = tf.random.uniform(shape=[], minval=-math.pi, maxval=math.pi)
#     image = tfa.image.rotate(image, rotate_angle)

#     # Add random noise
#     noise = tf.random.normal(shape=tf.shape(image), 
#                              mean=noise_mean, stddev=noise_std, 
#                              dtype=tf.float64)
#     image = image + noise
#     # Ensure that the image is still valid
#     image = tf.clip_by_value(image, 0.0, 1.0) 

#     return image, label


def create_weighted_cross_entropy(positive_weights):
    """
    Takes positive_weights as an argument and returns the 
    `weighted_cross_entropy_with_logits` function, which is used 
    in model compilation later. This way, the positive_weights are fixed 
    when creating the loss function, and the inner 
    `weighted_cross_entropy_with_logits` function correctly takes 
    only two arguments as expected by Keras.

    Inputs:
        1): positive_weights (Tensorflow): weight for positive examples 
                                           (i.e., minority class)

    Returns (function): the weighted_cross_entropy loss function 
    """

    def weighted_cross_entropy_with_logits(labels, logits):
        """
        Defines the weighted_cross_entropy loss function 
        to account for imbalance in labels of the dataset.

        Inputs:
            1) labels (Tensorflow): true labels
            2) logits (Tensorflow): predicted labels
        
        Returns (function): the weighted_cross_entropy loss function 
        """

        labels = tf.cast(labels, tf.float32)
        loss = tf.nn.weighted_cross_entropy_with_logits(
               labels, logits, positive_weights)
        return loss
    
    return weighted_cross_entropy_with_logits


def test_predict_eval(binary_prediction, true_labels, 
                      feature_num, class_num, labels):
    """
    Evaluate how the model predicts the labels from images oftest data.

    Inputs:
        binary_prediction (numpy array): model prediction of labels
        true_labels (pandas data frame): true labels from test data
        feature_num (int): total number of features
        class_num (int): total number of classes
        labels (lst): all possible features

    Returns: classifictation report (dic), confusion_matrix (numpy_array),

    """

    classifi_report = classification_report(true_labels, binary_prediction,
                                            target_names=labels)
    
    confusion_matrix = multilabel_confusion_matrix(true_labels, 
                                                   binary_prediction)

    # Compute ROC curve and ROC area for each class
    fpr = dict() # False Positive Rate
    tpr = dict() # True Positive Rate
    roc_auc = dict()

    plt.figure(figsize=(20, 20))
    for i in range(feature_num * class_num):
        # Plot conofusion matrix for each label
        plt.subplot(feature_num, class_num, i + 1)
        seaborn.heatmap(confusion_matrix[i], annot=True, 
                        fmt='g', cmap="Blues",
                        cbar=False)
        plt.title(labels[i])
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Compute ROC curve and area
        fpr[i], tpr[i], _ = roc_curve(true_labels.iloc[:, i], 
                                      binary_prediction[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.tight_layout()

    # Save the image
    plt.savefig('confusion_matrices.jpg', format='jpg', dpi=300)

    return classifi_report, confusion_matrix, roc_auc