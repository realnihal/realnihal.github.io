---
layout: post
title:  Food Vision Using Tensorflow
description: Using the power of mixed learning and tensorflow to make a Food Vision model.
date:   2021-08-11 15:01:35 +0300
image:  '/img/posts/foodvision/food_vision.jpg'
tags:   [Food, TensorFlow, Deep Learning, Computer Vision]
---

# **Food Vision Using an EffecientNetX and Tensorflow**
We are using the **Food101** standard database to create a deep learning model that can tell the difference and make predictions between 101 classes of food. using all 75,750 training images and 25,250 testing images. We will also use mixed learning to increase the speed of the training process.

Our goal is to beat the original [DeepFood ](https://arxiv.org/ftp/arxiv/papers/1606/1606.05675.pdf) paper. they have an accuracy of 77.4%.

We will use the power of transfer learning by incorporating the EfficientNetX architecture, using the following steps:

1.   Feature Extraction
2.   Fine Tuning

Here is the [Link to the complete github repository](https://github.com/realnihal/Food-Vision-Using-Tensorflow).

## Checking for the right GPU
Since we are planning to use mixed learning, we need a compatible GPU.
This model is being trained on Google's colab and they provides 3 types of free Nvidia GPU's.

1. K80(not compatible)
2. P100(not compatible)
3. Tesla T4(compatible)

Knowing this we need access to an Nvidia Tesla T4(from colab) or any GPU with a compute score of 7+ of our own.

Lets find out our GPU type by the following command.


```python
!nvidia-smi -L
```

    GPU 0: Tesla T4 (UUID: GPU-b54d7bc4-6911-4395-1619-170d04e3161d)


Great! we have a compatible GPU ie, the Tesla T4. If you do not have a compatible GPU try factory runtime -> factory reset to reset your session.



```python
# Show the Tensorflow Version (run this in Google Colab) 
import tensorflow as tf
print(tf.__version__)
```

    2.4.1


There is a known bug with tensorflow 2.5.0 specifically that does'nt work with mixed learning. To avoid that use the following code to downgrade to version 2.4.1.[For more info](https://github.com/tensorflow/tensorflow/issues/49725).


```python
# Downgrade Tensorflow Version (run this in Google Colab) 
!pip install tensorflow==2.4.1
```

## Importing the Data
We have created some helper functions that we are importing into our project.



```python
!wget https://raw.githubusercontent.com/realnihal/Random_Code/master/helper_functions.py

from helper_functions import create_tensorboard_callback, plot_loss_curves
```

    --2021-08-05 05:15:53--  https://raw.githubusercontent.com/realnihal/Random_Code/master/helper_functions.py
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 10510 (10K) [text/plain]
    Saving to: ‘helper_functions.py’
    
    helper_functions.py 100%[===================>]  10.26K  --.-KB/s    in 0s      
    
    2021-08-05 05:15:53 (89.1 MB/s) - ‘helper_functions.py’ saved [10510/10510]
    


The dataset **Food101** in available in to download from tensorflow datasets. You can find out more about this from here [Tensorflow Datasets](https://www.tensorflow.org/datasets)


```python
#importing tfds
import tensorflow_datasets as tfds
```


```python
#checking the name of the dataset in present within tfds
datasets_list = tfds.list_builders()
print("food101" in datasets_list)
```

    True


This may take a few minutes since the data is so large.



```python
#Loading the dataset
(train_data, test_data), ds_info = tfds.load(name='food101',
                                             split=['train','validation'],
                                             shuffle_files=True,
                                             as_supervised=True,
                                             with_info=True)
```

## Data exploration

lets see what the downloaded data contains. This allows us to get a more better idea of how to preprocess the data.


```python
#Features of Food101 from TFDS
ds_info.features
```

    FeaturesDict({
        'image': Image(shape=(None, None, 3), dtype=tf.uint8),
        'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=101),
    })



First thing we notice is that the image is in the **uint8** format we need to convert it into **float32**.


```python
#Printing a few class names
class_names = ds_info.features["label"].names
class_names[:10] #only print 20 to avoid spamming the output
```




    ['apple_pie',
     'baby_back_ribs',
     'baklava',
     'beef_carpaccio',
     'beef_tartare',
     'beet_salad',
     'beignets',
     'bibimbap',
     'bread_pudding',
     'breakfast_burrito']




```python
#Lets take one sample to dive deeper and view our data
train_one_sample = train_data.take(1)


#Printing the image features
for image, label in train_one_sample:
  print(f'''
  Image shape: {image.shape}
  Iimage datatype: {image.dtype}
  Target class from Food101 (tensor form): {label}
  Class name (str form): {class_names[label.numpy()]}
  ''')
```

    
      Image shape: (512, 512, 3)
      Iimage datatype: <dtype: 'uint8'>
      Target class from Food101 (tensor form): 25
      Class name (str form): club_sandwich
      



```python
#Checking whether the image is normalized
image
```




    <tf.Tensor: shape=(512, 512, 3), dtype=uint8, numpy=
    array([[[135, 156, 175],
            [125, 148, 166],
            [114, 136, 159],
            ...,
            [ 26,   5,  12],
            [ 26,   3,  11],
            [ 27,   4,  12]],
    
           [[128, 150, 171],
            [115, 140, 160],
            [102, 127, 149],
            ...,
            [ 28,   7,  14],
            [ 29,   6,  14],
            [ 30,   7,  15]],
    
           [[112, 139, 160],
            [ 99, 127, 148],
            [ 87, 115, 137],
            ...,
            [ 29,   6,  16],
            [ 31,   5,  16],
            [ 32,   6,  17]],
    
           ...,
    
           [[ 48,  47,  53],
            [ 53,  52,  58],
            [ 52,  51,  59],
            ...,
            [111,  99,  99],
            [108,  98,  97],
            [106,  96,  97]],
    
           [[ 44,  45,  47],
            [ 48,  49,  51],
            [ 46,  47,  51],
            ...,
            [108,  96,  98],
            [105,  94,  98],
            [102,  93,  96]],
    
           [[ 40,  42,  41],
            [ 45,  47,  46],
            [ 44,  45,  49],
            ...,
            [105,  95,  96],
            [104,  93,  99],
            [100,  91,  96]]], dtype=uint8)>




```python
tf.reduce_min(image), tf.reduce_max(image)
```




    (<tf.Tensor: shape=(), dtype=uint8, numpy=0>,
     <tf.Tensor: shape=(), dtype=uint8, numpy=255>)




```python
#Plotting an image from the dataset to check if our labels are correct
import matplotlib.pyplot as plt
plt.imshow(image)
plt.title(class_names[label.numpy()])
plt.axis(False)
```




    (-0.5, 511.5, 511.5, -0.5)

    
![png](\img\posts\foodvision\output_22_1.png)


## Preprocessing the data

from our initial data exploration we found that we need to do the following things:



1.   Reshaping our images to a standard size - [255,255]
2.   Converting our images into float32.
3.   Shuffling them again.
4.   Set up batches to ensure we dont run out of memory.



```python
# Make a function for preprocessing images
def preprocess_img(image, label, img_shape=224):
  """
  Converts image datatype from 'uint8' -> 'float32' and reshapes image to
  [img_shape, img_shape, color_channels]
  """
  image = tf.image.resize(image, [img_shape, img_shape]) # reshape to img_shape
  #image = image/255.0 (not required)
  return tf.cast(image, tf.float32), label # return (float32_image, label) tuple
```


```python
#Using our preprocess function to test on the sample image
preprocessed_img = preprocess_img(image, label)[0]
print(f"Image before preprocessing:\n {image[:2]}...,\nShape: {image.shape},\nDatatype: {image.dtype}\n")
print(f"Image after preprocessing:\n {preprocessed_img[:2]}...,\nShape: {preprocessed_img.shape},\nDatatype: {preprocessed_img.dtype}")
```

    Image before preprocessing:
     [[[135 156 175]
      [125 148 166]
      [114 136 159]
      ...
      [ 26   5  12]
      [ 26   3  11]
      [ 27   4  12]]
    
     [[128 150 171]
      [115 140 160]
      [102 127 149]
      ...
      [ 28   7  14]
      [ 29   6  14]
      [ 30   7  15]]]...,
    Shape: (512, 512, 3),
    Datatype: <dtype: 'uint8'>
    
    Image after preprocessing:
     [[[122.83163   146.17346   165.81633  ]
      [ 95.07653   122.122444  144.47958  ]
      [ 72.5051    106.994896  134.34694  ]
      ...
      [ 20.714308    2.3570995   3.9285717]
      [ 27.285715    6.285714   13.285714 ]
      [ 28.28575     5.2857494  13.285749 ]]
    
     [[ 88.65305   119.41326   140.41327  ]
      [ 74.59694   108.30102   133.02042  ]
      [ 75.2551    112.57143   141.91325  ]
      ...
      [ 26.857143    6.285671   11.040798 ]
      [ 30.061235    6.86222    16.795908 ]
      [ 31.688843    5.688843   16.688843 ]]]...,
    Shape: (224, 224, 3),
    Datatype: <dtype: 'float32'>


The data.AUTOTUNE function and the prefetch function work in tandem to utilize the multiple cores available to us and the gpu to efficiently process all of our images as quickly as possible.


```python
# Map preprocessing function to training data (and paralellize)
train_data = train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
# Shuffle train_data and turn it into batches and prefetch it (load it faster)
train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

# Map prepreprocessing function to test data
test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
# Turn test data into batches (don't need to shuffle)
test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)
```


```python
train_data, test_data
```




    (<PrefetchDataset shapes: ((None, 224, 224, 3), (None,)), types: (tf.float32, tf.int64)>,
     <PrefetchDataset shapes: ((None, 224, 224, 3), (None,)), types: (tf.float32, tf.int64)>)



## Impementing Mixed Precision
Now lets implement Mixed precision. Here we can try to use flast 16 on some layers to improve speed and efficiency. It is only compatible with GPU's with a compute score of 7+.


```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
```

    INFO:tensorflow:Mixed precision compatibility check (mixed_float16): OK
    Your GPU will likely run quickly with dtype policy mixed_float16 as it has compute capability of at least 7.0. Your GPU: Tesla T4, compute capability 7.5


    INFO:tensorflow:Mixed precision compatibility check (mixed_float16): OK
    Your GPU will likely run quickly with dtype policy mixed_float16 as it has compute capability of at least 7.0. Your GPU: Tesla T4, compute capability 7.5


## Defining our Feature-Extraction model
Its time to define our model. Here is the order in which we define our layers:

1.   InputLayer - takes input as an image[float 32]
2.   EfficientNetB0 - this is the main brains of our feature extraction
3.   Pooling Layer - to convert the output of efficient net into a feature vector
4.   Output Layer - gives an output as a probability distribution.

We are ensuring that the layers of the EfficientNet is Freezed. This is to prevent any deviation of the already learned patterns. We need to also specify the float32 as output to the activation layer, as using the global float16 may cause issues with the softmax activation function.




```python
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Create base model
input_shape = (224, 224, 3)
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False # freeze base model layers

# Create Functional model 
inputs = layers.Input(shape=input_shape, name="input_layer")
# x = preprocessing.Rescaling(1./255)(x) EfficientNetBX models have rescaling built-in but if your model didn't you could have a layer like below
x = base_model(inputs, training=False) # set base_model to inference mode only
x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
x = layers.Dense(len(class_names))(x) # want one output neuron per class 
# Separate activation of output layer so we can output float32 activations
outputs = layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x) 
model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", # Use sparse_categorical_crossentropy when labels are *not* one-hot
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])
```

    Downloading data from https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5
    16711680/16705208 [==============================] - 0s 0us/step



```python
#Printing a summary of the model
model.summary()
```

    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_layer (InputLayer)     [(None, 224, 224, 3)]     0         
    _________________________________________________________________
    efficientnetb0 (Functional)  (None, None, None, 1280)  4049571   
    _________________________________________________________________
    pooling_layer (GlobalAverage (None, 1280)              0         
    _________________________________________________________________
    dense (Dense)                (None, 101)               129381    
    _________________________________________________________________
    softmax_float32 (Activation) (None, 101)               0         
    =================================================================
    Total params: 4,178,952
    Trainable params: 129,381
    Non-trainable params: 4,049,571
    _________________________________________________________________



```python
#Checking the trainability and datatypes of the layers
for layer in model.layers:
  print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)
```

    input_layer True float32 <Policy "float32">
    efficientnetb0 False float32 <Policy "mixed_float16">
    pooling_layer True float32 <Policy "mixed_float16">
    dense True float32 <Policy "mixed_float16">
    softmax_float32 True float32 <Policy "float32">


Creating our callback functions.

1. Tensorboard callback to save our training data.
2. Model Checkpoint to save our best model.
 


```python
# Create TensorBoard callback (already have "create_tensorboard_callback()" from a previous notebook)
from helper_functions import create_tensorboard_callback

# Create ModelCheckpoint callback to save model's progress
checkpoint_path = "model_checkpoints/cp.ckpt" # saving weights requires ".ckpt" extension
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      monitor = 'val_accuracy', # save the model weights with best validation accuracy
                                                      save_best_only=True, # only save the best weights
                                                      save_weights_only=True, # only save model weights (not whole model)
                                                      verbose=1)
```

## Fitting the data on the Feature-Extraction model.

We are using the early_stopping callback to prevent any major overfitting and running the training for 3 epochs. We are testing on only 15 percent of the data to save time. Yet this is going to take a while processing over 100,000 images per epoch. So, I'm going to grab a cup of coffee.


```python
history_101_food_classes_feature_extract = model.fit(train_data, 
                                                     epochs=3,
                                                     steps_per_epoch=len(train_data),
                                                     validation_data=test_data,
                                                     validation_steps=int(0.15 * len(test_data)),
                                                     callbacks=[create_tensorboard_callback("training_logs", 
                                                                                            "efficientnetb0_101_classes_all_data_feature_extract"),
                                                                model_checkpoint])
```

    Saving TensorBoard log files to: training_logs/efficientnetb0_101_classes_all_data_feature_extract/20210805-060548
    Epoch 1/3
    2368/2368 [==============================] - 192s 80ms/step - loss: 0.6993 - accuracy: 0.8149 - val_loss: 1.1099 - val_accuracy: 0.7021
    
    Epoch 00001: val_accuracy did not improve from 0.71478
    Epoch 2/3
    2368/2368 [==============================] - 167s 70ms/step - loss: 0.6780 - accuracy: 0.8221 - val_loss: 1.1195 - val_accuracy: 0.7058
    
    Epoch 00002: val_accuracy did not improve from 0.71478
    Epoch 3/3
    2368/2368 [==============================] - 171s 72ms/step - loss: 0.6579 - accuracy: 0.8262 - val_loss: 1.1418 - val_accuracy: 0.6989
    
    Epoch 00003: val_accuracy did not improve from 0.71478


## Viewing the results, Saving our Feature-Extraction model.




```python
#Evaluating our model on the test data
results_feature_extract_model = model.evaluate(test_data)
results_feature_extract_model
```

    790/790 [==============================] - 55s 70ms/step - loss: 1.1396 - accuracy: 0.7001





    [1.1395542621612549, 0.7001188397407532]



We got an accuracy of about 70% its close to our target ie [Deepfood's](https://arxiv.org/ftp/arxiv/papers/1606/1606.05675.pdf) original score of 77.4%. We still have another step to do, lets hope for the best.


```python
#Saving our model
model.save("101_food_feature_extract_mixedpred_model")
```

    INFO:tensorflow:Assets written to: 101_food_feature_extract_mixedpred_model/assets


    INFO:tensorflow:Assets written to: 101_food_feature_extract_mixedpred_model/assets



```python
#Loading our model into a seperate model
loaded_model = tf.keras.models.load_model("101_food_feature_extract_mixedpred_model")
loaded_model.load_weights(checkpoint_path)
```


```python
#Checking if our layer types are accurate in loaded_model
for layer in loaded_model.layers:
  print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)
```

    input_layer True float32 <Policy "float32">
    efficientnetb0 True float32 <Policy "mixed_float16">
    pooling_layer True float32 <Policy "mixed_float16">
    dense True float32 <Policy "mixed_float16">
    softmax_float32 True float32 <Policy "float32">



```python
#Evaulating on our loaded_model
test_eval = loaded_model.evaluate(test_data)
```

    790/790 [==============================] - 49s 60ms/step - loss: 1.0722 - accuracy: 0.7116



```python
#Seeing if the results match our actual model.
results_feature_extract_model == test_eval
```




    False



## Making our Fine-Tuning model

To fine tune the model lets unfreeze the layers in our EfficientNet.


```python
# Are any of the layers in our model frozen?
for layer in loaded_model.layers:
  layer.trainable = True # set all layers to trainable
  print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy) # make sure loaded model is using mixed precision dtype_policy ("mixed_float16")
```

    input_layer True float32 <Policy "float32">
    efficientnetb0 True float32 <Policy "mixed_float16">
    pooling_layer True float32 <Policy "mixed_float16">
    dense True float32 <Policy "mixed_float16">
    softmax_float32 True float32 <Policy "float32">


Setting up Some more Callbacks for Fine-tuning

1. Tensorboard callback - to save our training data.
2. Model Checkpoint - to save our best model.
3. Early Stopping - to stop out training if our model is not improving



```python
# Setup EarlyStopping callback to stop training if model's val_loss doesn't improve for 3 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=3) # if val loss decreases for 3 epochs in a row, stop training

# Create ModelCheckpoint callback to save best model during fine-tuning
checkpoint_path = "fine_tune_checkpoints/"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      save_best_only=True,
                                                      monitor="val_loss")

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=2,
                                                 verbose=1, # print out when learning rate goes down 
                                                 min_lr=1e-7)
```


```python
loaded_model.compile(loss="sparse_categorical_crossentropy", # sparse_categorical_crossentropy for labels that are *not* one-hot
                        optimizer=tf.keras.optimizers.Adam(0.0001), # 10x lower learning rate than the default
                        metrics=["accuracy"])
```

## Fitting the data on our Fine-Tuning model.

We are running the training for 100 epochs. The training will be stopped when the early stopping function is called. This is again going to take a while processing over 100,000 images per epoch. So, I'm going to grab another cup of coffee.


```python
history_101_food_classes_all_data_fine_tune = loaded_model.fit(train_data,
                                                        epochs=100, # fine-tune for a maximum of 100 epochs
                                                        steps_per_epoch=len(train_data),
                                                        validation_data=test_data,
                                                        validation_steps=int(0.15 * len(test_data)), # validation during training on 15% of test data
                                                        callbacks=[create_tensorboard_callback("training_logs", "efficientb0_101_classes_all_data_fine_tuning"), # track the model training logs
                                                                   model_checkpoint, # save only the best model during training
                                                                   early_stopping, # stop model after X epochs of no improvements
                                                                   reduce_lr]) # reduce the learning rate after X epochs of no improvements
```

    Saving TensorBoard log files to: training_logs/efficientb0_101_classes_all_data_fine_tuning/20210805-063310
    Epoch 1/100
    2368/2368 [==============================] - 313s 127ms/step - loss: 0.8116 - accuracy: 0.7748 - val_loss: 0.8895 - val_accuracy: 0.7524
    INFO:tensorflow:Assets written to: fine_tune_checkpoints/assets


    INFO:tensorflow:Assets written to: fine_tune_checkpoints/assets


    Epoch 2/100
    2368/2368 [==============================] - 296s 122ms/step - loss: 0.4660 - accuracy: 0.8666 - val_loss: 0.9664 - val_accuracy: 0.7505
    Epoch 3/100
    2368/2368 [==============================] - 290s 122ms/step - loss: 0.2495 - accuracy: 0.9273 - val_loss: 1.0183 - val_accuracy: 0.7532
    
    Epoch 00003: ReduceLROnPlateau reducing learning rate to 1.9999999494757503e-05.
    Epoch 4/100
    2368/2368 [==============================] - 290s 122ms/step - loss: 0.0796 - accuracy: 0.9808 - val_loss: 1.0898 - val_accuracy: 0.7773


## Viewing the results, Saving our Fine-Tuning model.



```python
loaded_model.save("efficientnetb0_fine_tuned_101_classes_mixed_precision")
```

    INFO:tensorflow:Assets written to: efficientnetb0_fine_tuned_101_classes_mixed_precision/assets


    INFO:tensorflow:Assets written to: efficientnetb0_fine_tuned_101_classes_mixed_precision/assets



```python
final_score = loaded_model.evaluate(test_data)
```

    790/790 [==============================] - 56s 70ms/step - loss: 1.0577 - accuracy: 0.7834


Yes! we did it! we beat the original [DeepFood's](https://arxiv.org/ftp/arxiv/papers/1606/1606.05675.pdf) score of 77.4%.


```python
#saving the tensorboard data online - I have multiple models since Iyes tried a couple of ideas
!tensorboard dev upload --logdir ./training_logs \
  --name "Fine-tuning EfficientNetB0 on all Food101 Data" \
  --description "Training results for fine-tuning EfficientNetB0 on Food101 Data with learning rate 0.0001" \
  --one_shot
```

[Click here to view the analysis at tensorboard.](https://tensorboard.dev/experiment/5coYXuxLQdyzAQAtJ0Nm8g/)

Downloading the files to my computer. these files should be available in the github repository.


```python
!zip -r /content/file.zip '101_food_feature_extract_mixedpred_model'
```

      adding: 101_food_feature_extract_mixedpred_model/ (stored 0%)
      adding: 101_food_feature_extract_mixedpred_model/assets/ (stored 0%)
      adding: 101_food_feature_extract_mixedpred_model/variables/ (stored 0%)
      adding: 101_food_feature_extract_mixedpred_model/variables/variables.index (deflated 73%)
      adding: 101_food_feature_extract_mixedpred_model/variables/variables.data-00000-of-00001 (deflated 8%)
      adding: 101_food_feature_extract_mixedpred_model/saved_model.pb (deflated 92%)



```python
from google.colab import files
files.download("/content/file.zip")
```


```python
!zip -r /content/file2.zip 'efficientnetb0_fine_tuned_101_classes_mixed_precision'
```

      adding: efficientnetb0_fine_tuned_101_classes_mixed_precision/ (stored 0%)
      adding: efficientnetb0_fine_tuned_101_classes_mixed_precision/assets/ (stored 0%)
      adding: efficientnetb0_fine_tuned_101_classes_mixed_precision/variables/ (stored 0%)
      adding: efficientnetb0_fine_tuned_101_classes_mixed_precision/variables/variables.index (deflated 78%)
      adding: efficientnetb0_fine_tuned_101_classes_mixed_precision/variables/variables.data-00000-of-00001 (deflated 8%)
      adding: efficientnetb0_fine_tuned_101_classes_mixed_precision/saved_model.pb (deflated 92%)



```python
files.download("/content/file2.zip")
```

## Conclusions

We acheived an accuracy of about 78.3%. With this we beat the score set by the original [DeepFood](https://arxiv.org/ftp/arxiv/papers/1606/1606.05675.pdf) paper.

It feels great that we achieved our goal. But for those who were keen, our training accuracy was very high compared to our test accuracy. This is probably due to over fitting. On investigation I found out that the [EfficientNet](https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet) model was trainied upon the [ImageNet](https://www.image-net.org/index.php) dataset. These images closely resemble our images. due to this our model ended up slightly overfitting.

**Nevertheless, we acheived our objective and that's all that matters.**

Feel free to contact me on any of my social's for anything!, Peace out!
