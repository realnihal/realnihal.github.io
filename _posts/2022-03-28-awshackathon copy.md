---
layout: post
title:  Our prize winning AWS Hackathon entry
description: explaining the project that won runner up Computer Vision award.
date:   2022-03-28 15:01:35 +0300q
image:  '/img/posts/aws/aws.jpg'
tags:   [TensorFlow, Computer Vision, Deep Learning]
---

# AWS - Hackathon

This article is written to explain my project that I submitted to the AWS Hackathon on Devpost. Me and my friend had initial started the project as a learning step to build and deploy a machine learning project. We eventually ended up winning the runner up for Computer Vision for our efforts.

# About the project - Airken

## Inspiration

With rapid urbanization and industrialization Air pollution has become an alarming issue globally. Among different air pollutants particles , matter in (PM2.5) size range are able traverse deeply into our respiratory tract , transmitting hazardous chemicals into the human lungs and blood causing respiratory and cardiovascular health issues . We aim to develop image-based air quality analysis application , in particular a one that estimates the concentration of particulate matter with diameters in range of 2.5 micrometers.

## What it does

It can tell you what your breathing from images generated using consumer grade hardware. we have used a deep Convolutional Neural Network (CNN) with the high computational power of Amazon EC2 DL1 instance to solve a computer vision problem of classifying natural images into 3 different categories ;healthy , moderate and danger based on their PM2.5 concentrations.

## How we built it

We created our own dataset by combining data from 2 sources . first , we took data from an existing dataset that has various dated images of phoenix city from the year 2014 . Then we obtained the corresponding Pm 2.5 values for each day from historical data available on [aqicn.org](http://aqicn.org/) and manually noted in them in an excel sheet . we then separated the images into three classes: Healthy (<30), Moderate 30~54), and Danger (>54)

We learned that, some reasons for overfitting could be high imbalance in data , we had 2232 healthy images where as the moderate and danger categories had only 1480 and 591 images respectively . so we have written a code to randomly oversample images in these folders until we had 2232. Then we trained our model following the same method as before but this time we used ImageNet weights un-freezed the last 100 layers for training , then fine-tuned the model with a lower learning rate ,we also used early stopping callback with a patience of 3 . Now, there are some more things we have done to improve our model. We migrated our model onto the dl1 instance and initially trained using 1 gaudi processor. Later,we were able to parallelize our workflow into 8 gaudi processors using some helpful documentation available on [habana.ai](http://habana.ai/). This significantly improved our performance and efficiency .We were able to quickly test , experiment and tweak changes . finally achieved an accuracy of about 76 %. We then converted our model into a tensorflow lite model and used a huggingface demo tool called gradio to quickly demonstrate our application.

## Challenges we ran into

Initially as we tried training our model we observed some of the image files in the dataset were corrupted .We automated the process of scanning through and deleting these files and then proceeded to training model . We used EfficientNet architecture since its renown for its performance in low spec machines, this is called transfer learning . initially we freezed all the trainable layers except the last 30 layers and set the weights parameter to false. once we completed training we fine tuned our model by setting a lower learning rate and unfreezing all the layers . we achieved a validation accuracy of about 60 % but our training accuracy was high, the model could be overfitting.

## Accomplishments that we're proud of

As we have seen we were able an accuracy of about 76 % which is pretty good considering the data we had. the model will be able to perform better if we feed a bigger and precise data. Now , why do we need this application and how is it different from the existing air monitoring system systems, we already know Exposure to fine particles can cause long term health effects such as lung and heart problem but they also cause short term health effects such as eye, nose, throat irritation. The existing air quality monitoring methods are highly dependent on monitoring stations, which are again situated far away because of the high setup cost and expensive equipment. also , air quality can change from one hour to the next. For a specific location, the air quality depends on how air moves through the area and how people are influencing the air , so we cannot rely on something that was predicted a while back or in some place that is not our site of interest .This application helps you analyse the air quality around you with a single picture and more importantly is portable. We can carry it anywhere we wish to and know what your being exposed to .

## What we learned

- Transfer learning
- dealing with imbalanced datasets
- dealing with overfitted models

## What's next for Airken

### In App features:

Honestly I don’t know, I think there is potential for this app to integrate into our routine. People might want features to help plan their day out, integrate with their phone’s calender and suggest the best time to plan an outdoor activity.

### Azure integration:

### AI development:

Right now the Artificial intelligence that runs the machine runs on the azure cloud. In the future we want to be able to bring it into the app and run it natively.

### Built with:

1. TensorFlow, TFlite
2. Custom Built Datasets
3. Flutter, Material Design
4. Azure, Azure AI and Data Services
5. Python, Flask, Nginx, Gunicorn
6. Postman, OpenWeather APIs