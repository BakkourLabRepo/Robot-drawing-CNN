####################
#                  #
#  Import modules  #
#                  #
####################
from Helper_Functions import *
import tensorflow as tf
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
#import matplotlib.pyplot as plt
#import sys
import contextlib
import argparse

#########################################
#                                       #
#  Define command-line arguments used   #
#                                       #
#########################################
parser = argparse.ArgumentParser(description='Process some inputs.')
parser.add_argument('core', type=str, help='Modality: CPU vs GPU')
parser.add_argument('--environment', type=str, choices=["local", "midway"], required=True, 
                    help="Specify 'local' for running on a local machine or 'midway' for running on the Midway cluster.")
parser.add_argument('--model_name', type=str, choices=["base", "VGG19"], required=True, help='The name of the model')

args = parser.parse_args()

# Extract command-line arguments
core = args.core
environment = args.environment
model_name = args.model_name

#########################################
#                                       #
# Define configuration and usage of GPU #
#                                       #
#########################################
# Partition
# partition = sys.argv[2]

# Configure GPUs or CPU
physical_gpus = tf.config.list_physical_devices('GPU')
if physical_gpus:
    try:
        # Set memory growth to true for all GPUs
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Error handling
        print(e)

# Initialize a distribution strategy for both GPU and CPU
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
FEATURE = ["bo", "he", "ar", "le", "an"]
FEATURE_NUM = len(FEATURE)
# Number of options per class (e.g., 6 possible arms and legs for each robot)
CLASS_NUM = 6
# All potential "FEATURE * CLASS_NUM" combinations of lables
LABELS = [f"{f}_{n}" for f in FEATURE for n in range(1, CLASS_NUM + 1)]

# Dataset spliting for model training, validation, and testing
SPLIT_RATIO = (0.8, 0.1, 0.1)
# An integer for reproducibility of data spliting using sklearn.model_selection
RANDOM_STATE = 42 

# Dropout rate
DROPOUT_RATE = 0.4

# Setting a learning rate
BASE_LEARNING_RATE = 0.01

# A threshold value to decide which class the predicted probability points to
# This threshold should take imbalanced 0s and 1s into account
P_R_THRESHOLD = 0.3

# Class weights to account for imbalanced dataset
TOTAL_FEATURES = (CLASS_NUM * FEATURE_NUM)
# one_weight = (1 / FEATURE_NUM) * (TOTAL_FEATURES / 2.0)
# zero_weight = (1 / (TOTAL_FEATURES - FEATURE_NUM)) * (TOTAL_FEATURES / 2.0)
POSITIVE_WEIGHT_BODY = 6
POSITIVE_WEIGHT_OTHER = 7 # higher because category can be missing
POSITIVE_WEIGHTS = [POSITIVE_WEIGHT_BODY]*CLASS_NUM 
POSITIVE_WEIGHTS += [POSITIVE_WEIGHT_OTHER]*(TOTAL_FEATURES - CLASS_NUM) 
POSITIVE_WEIGHTS = tf.constant(POSITIVE_WEIGHTS, dtype=tf.float32)

# Model checkpoint directory
CHECKPOINT_DIR = "ModelCheckpoint"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# SavedModel directory
SAVEDMODEL_DIR = "SavedModel"
os.makedirs(SAVEDMODEL_DIR, exist_ok=True)

#################
#               #
#  Load images  #
#               #
#################
# Define the `robots_stim_path` path storing robot drawings
if environment == "midway":
    project_path = "/project/bakkour/projects/feat_predict/"
    robots_stim_path = os.path.join(project_path, "robots", "stim/")
elif environment == "local":
    robots_stim_path = "combs/"

image_array, label_df = images_and_labels(robots_stim_path, FEATURE_NUM, CLASS_NUM, LABELS)
print(f"The shape of the image array is {image_array.shape}")
print("-----------------------------------------------------------------------")
print("The first five rows of FEATURE vectors dataframe")
print(label_df.head())
print("-----------------------------------------------------------------------")


#####################
#                   #
#  Preprocess data  #
#                   #
#####################
train_data, val_data, test_data = preprocessing(image_array, label_df, 
                                                SPLIT_RATIO, RANDOM_STATE,
                                                batch_size)

print("Diplay the first batch of images alongside their associated LABELS")
image_batch, label_batch = next(iter(train_data.take(1)))

# Number of images to display from the batch
num_images = len(image_batch)
"""
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
        string += (LABELS[i] + "_")
    plt.title(string[:-1])
    plt.axis('off')
"""
print("-----------------------------------------------------------------------")


#################################
#                               #
#  Build and compile the model  #
#                               #
#################################
# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', 
                                patience=3, verbose=1, mode='min', 
                                restore_best_weights=True)

# Define checkpoint callback
model_checkpoint_callback = ModelCheckpoint(
    filepath=f'{CHECKPOINT_DIR}/{model_name}/model-{{epoch:02d}}-{{val_loss:.2f}}.h5',
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1)

with strategy.scope():
    # Reload last checkpoint and epoch in case of interruption
    last_checkpoint, last_epoch = find_last_checkpoint(CHECKPOINT_DIR)

    if last_checkpoint:
        model = load_model(last_checkpoint)  # Load the last model
        print(f"Resuming training from epoch {last_epoch + 1}")
    else:
        # Create and compile a new model
        print("Starting training from scratch.")
        model = compile_model(model = "base", 
                              dropout_rate=DROPOUT_RATE, 
                              p_r_threshold=P_R_THRESHOLD, 
                              positive_weights=POSITIVE_WEIGHTS, 
                              base_learning_rate=BASE_LEARNING_RATE)
        last_epoch = 0
    
    model.fit(train_data, epochs=10, initial_epoch=last_epoch,
              validation_data=val_data, 
              callbacks=[early_stopping, model_checkpoint_callback])

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

        print("Use this model to predict LABELS from test_data images", file=f)
        predictions = model.predict(test_data)
        binary_prediction = np.where(predictions > P_R_THRESHOLD, 1, 0)
        _, _, true_labels = data_split(label_df, SPLIT_RATIO, 
                                       RANDOM_STATE=RANDOM_STATE)
        print("---------------------------------------------------------------", 
              file=f)

        print("Evaluate the prediction", file=f)
        classifi_report, confusion_matrix, roc_auc = \
            test_predict_eval(binary_prediction, true_labels, FEATURE_NUM, 
                              CLASS_NUM, LABELS)

        print(classifi_report, file=f)
        print("---------------------------------------------------------------", 
              file=f)

# Save predicted and true LABELS for robots on testing data
binary_prediction = pd.DataFrame(binary_prediction, columns=LABELS)
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
model.save(f'{SAVEDMODEL_DIR}/{model_name}')


# To load the whole model:
# model = load_model(f'{SAVEDMODEL_DIR}/{model_name}')