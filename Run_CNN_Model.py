####################
#                  #
#  Import modules  #
#                  #
####################
from Helper_Functions import *
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import Recall, Precision, BinaryAccuracy
from keras.models import load_model
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import contextlib


#########################################
#                                       #
# Define configuration and usage of GPU #
#                                       #
#########################################

import sys
import tensorflow as tf

# Modality (CPU vs GPU)
core = sys.argv[1]
# Partition
# partition = sys.argv[2]

if core == "gpu":
    # Initialize the SlurmClusterResolver
    resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    
    # Configure GPUs
    physical_gpus = tf.config.list_physical_devices('GPU')
    if physical_gpus:
        try:
            # Set memory growth to true for all GPUs
            for gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(physical_gpus), "Physical GPU(s),", len(logical_gpus), "Logical GPU(s)")
        except RuntimeError as e:
            print(e)

    # Initialize a distribution strategy
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(cluster_resolver=resolver)
else:
    strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Adjust batch size based on the number of replicas
base_batch_size = 32 # Base batch size
base_save_freq = 2 # Base save frequency
batch_size = base_batch_size * strategy.num_replicas_in_sync

# Printing detected devices (CPU / GPU)
print("Detected devices:")
print(tf.config.list_physical_devices())

# Save frequency (number of epochs * batch_size)
save_frequency = base_save_freq * batch_size

#########################################
#                                       #
#  Define parameters of the robot task  #
#                                       #
#########################################
# Body parts analyzed (e.g., arms + head + legs + antenna + feet)
feature = ["bo", "he", "ar", "le", "an"]
feature_num = len(feature)
# Number of options per class (e.g., 6 possible arms and legs for each robot)
class_num = 6
# All potential "feature * class_num" combinations of lables
labels = [f"{f}_{n}" for f in feature for n in range(1, class_num + 1)]
label_num = len(labels)

# Dataset spliting for model training, validation, and testing
split_ratio = (0.8, 0.1, 0.1)
# An integer for reproducibility of data spliting using sklearn.model_selection
random_state = 42 

# Dropout rate
dropout_rate = 0.4

# Setting a learning rate
base_learning_rate = 0.01

# A threshold value to decide which class the predicted probability points to
# This threshold should take imbalanced 0s and 1s into account
p_r_threshold = 0.3

# Class weights to account for imbalanced dataset
total_features = (class_num * feature_num)
one_weight = (1 / feature_num) * (total_features / 2.0)
zero_weight = (1 / (total_features - feature_num)) * (total_features / 2.0)
positive_weight_body = 6
positive_weight_other = 7 # higher because category can be missing
positive_weights = [positive_weight_body]*class_num 
positive_weights += [positive_weight_other]*(total_features - class_num) 
positive_weights = tf.constant(positive_weights, dtype=tf.float32)

#################
#               #
#  Load images  #
#               #
#################
# Store the directory that stores stimulus in the midway environment
project_path = "/project/bakkour/projects/feat_predict/"
# robots_stim_path = os.path.join(project_path, "robots", "stim/")
robots_stim_path = "combs"

image_array, label_df = images_and_labels(robots_stim_path, feature_num,class_num, labels)
print(f"The shape of the image array is {image_array.shape}")
print("-----------------------------------------------------------------------")
print("The first five rows of feature vectors dataframe")
print(label_df.head())
print("-----------------------------------------------------------------------")


#####################
#                   #
#  Preprocess data  #
#                   #
#####################
train_data, val_data, test_data = preprocessing(image_array, label_df, 
                                                split_ratio, random_state,
                                                batch_size)

print("Diplay the first batch of images alongside their associated labels")
image_batch, label_batch = next(iter(train_data.take(1)))

# Number of images to display from the batch
num_images = len(image_batch)

plt.figure(figsize=(16, 16))
for i in range(num_images):
    plt.subplot(num_images // 4, 4, i + 1)
    # Display the image
    plt.imshow(image_batch[i].numpy().squeeze(), cmap='gray')

    # Associate the image with the label
    feature_vec = label_batch[i].numpy()
    feature_idx = np.where(feature_vec == 1)[0].tolist()
    string = ""
    for i in feature_idx:
        string += (labels[i] + "_")
    plt.title(string[:-1])
    plt.axis("off")
print("-----------------------------------------------------------------------")


#################################
#                               #
#  Build and compile the model  #
#                               #
#################################

# Build a simple CNN (using 5*5 filters)
with strategy.scope():
    model = Sequential([
        Conv2D(32, (5, 5), activation='relu', input_shape=(750, 750, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),  # Additional Conv layer
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256, activation='relu'),  # Increased units
        Dropout(dropout_rate), 
        Dense(128, activation='relu'),
        Dropout(dropout_rate),  
        Dense(30, activation='sigmoid')
    ])

    # Compile the CNN model
    recall = Recall(thresholds=p_r_threshold)
    precision = Precision(thresholds=p_r_threshold)
    accuracy = BinaryAccuracy(threshold=p_r_threshold)

    model.compile(loss=create_weighted_cross_entropy(positive_weights),
                optimizer=Adam(learning_rate=base_learning_rate),
                metrics=[recall, precision, accuracy])

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', 
                                patience=3, verbose=1, mode='min', 
                                restore_best_weights=True)
    
    model.fit(train_data, epochs=20, 
          validation_data=val_data, callbacks=[early_stopping])

##########################################
#                                        #
#           Evaluate the model           #
#                                        #
##########################################

# Save model performance to a txt file
with open("Model Performance.txt", "w") as f:
    with contextlib.redirect_stdout(f):
        print("Fit the CNN model", file=f)
        print("---------------------------------------------------------------", 
              file=f)

        print("Fit on validation data", file=f)
        val_loss, val_recall, val_precision, val_accuracy = \
            model.evaluate(val_data)
        print(f"Validation data loss: {val_loss}", file=f)
        print(f"Validation data accuracy: {val_accuracy}", file=f)
        print(f"Validation data recall: {val_recall}", file=f)
        print(f"Validation data precision: {val_precision}", file=f)
        print("---------------------------------------------------------------", 
              file=f)

        print("Fit on testing data", file=f)
        test_loss, test_recall, test_precision, test_accuracy = \
            model.evaluate(test_data)
        print(f"Test data loss: {test_loss}", file=f)
        print(f"Test data accuracy: {test_accuracy}", file=f)
        print(f"Test data recall: {test_recall}", file=f)
        print(f"Test data precision: {test_precision}", file=f)
        print("---------------------------------------------------------------", 
              file=f)

        print("Use this model to predict labels from test_data images", file=f)
        predictions = model.predict(test_data)
        binary_prediction = np.where(predictions > p_r_threshold, 1, 0)
        _, _, true_labels = data_split(label_df, split_ratio, 
                                    random_state=random_state)
        print("---------------------------------------------------------------", 
              file=f)

        print("Evaluate the prediction", file=f)
        classifi_report, confusion_matrix, roc_auc = \
            test_predict_eval(binary_prediction, true_labels, feature_num, 
                            class_num, labels)

        print(classifi_report, file=f)
        print("---------------------------------------------------------------", 
              file=f)

# Save predicted and true labels for robots on testing data
binary_prediction = pd.DataFrame(binary_prediction, columns=labels)
predicted_labels_string = binary_prediction.apply(\
    lambda x: "_".join(list(np.array(binary_prediction.columns)[x == 1])), axis=1)
true_labels_string = true_labels.apply(\
    lambda x: "_".join(list(np.array(true_labels.columns)[x == 1])), axis=1)
testing_data_prediction = pd.DataFrame(list(zip(predicted_labels_string, 
                                                true_labels_string)),
                                       columns=["Predicted", "True"])
# Save DataFrame to a text file, using comma as the delimiter
testing_data_prediction.to_csv('testing_data_prediction.csv', index=False)


####################
#                  #
#  Save the model  #
#                  #
####################
model.save('test_model')
# model = load_model('test_model')