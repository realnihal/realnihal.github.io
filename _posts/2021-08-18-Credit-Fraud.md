---
layout: post
title:  Credit Card Fraud Detection
description: Labeling anonymized credit card transactions as fraudulent or genuine.
date:   2021-08-18 15:01:35 +0300
image:  '/img/posts/creditfraud/creditfraud.jpeg'
tags:   [Machine Learning, Lifestyle]
---


## Context and Dataset information

It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

The dataset that we are using is obtained from the [**Kaggle dataset by the Machine Learning Group - ULB**](https://www.kaggle.com/mlg-ulb/creditcardfraud).

The dataset that we are using contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. **This dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions**.
It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, they cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. **Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise**.

[**To check out the complete code for the project click here**](https://github.com/realnihal/Credit-Card-Fraud-Problem).


```python
df.Class.value_counts()
```
    0    284315
    1       492
    Name: Class, dtype: int64


This is interesting! the dataset is imbalanced. 
In classification machine learning problems(binary and multiclass), datasets are often imbalanced which means that one class has a higher number of samples than others. This will lead to bias during the training of the model, the class containing a higher number of samples will be preferred more over the classes containing a lower number of samples. Having bias will, in turn, increase the true-negative and false-positive rates (ie, the precision and recall).

Let's see the results without adjusting for the imbalanced bias on a base-model. I have used a **simple logistic regression model** for this.

## Base-Model

```python
log_class = LogisticRegression()
grid = {'C': 10.0 ** np.arange(-2, 3), 'penalty': ['l1', 'l2']}
cv = KFold(n_splits=5, shuffle=False, random_state=None)
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
```


```python
clf = GridSearchCV(log_class, grid, cv=cv, n_jobs=-1, scoring='f1_macro')
clf.fit(X_train, y_train)
```
Let's find the confusion matrix and the accuracy.

```
    [[198952     69]
     [   111    233]]

    0.9990971333985403
```
Looks like we got an accuracy of over 99 percent! But wait a minute we have an issue. The confusion matrix looks off... lets look at the classification report.

```
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00    199021
               1       0.77      0.68      0.72       344
    
        accuracy                           1.00    199365
       macro avg       0.89      0.84      0.86    199365
    weighted avg       1.00      1.00      1.00    199365
```  

The precision of the failed cases is about 77 percent, not bad but I guess we can do better. Lets try a more complex model

## RandomForestClassifier


```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(class_weight=class_weight)
clf.fit(X_train, y_train)
```

```python
prediction = clf.predict(X_test)
print(confusion_matrix(y_test, prediction))
print(accuracy_score(y_test, prediction))
print(classification_report(y_test, prediction))
```

    [[199006     15]
     [    94    250]]
    0.9994532641135605


Looks like we got an accuracy of over 99 percent! But the recall is still too high. We can do better!

```
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00    199021
               1       0.94      0.73      0.82       344
    
        accuracy                           1.00    199365
       macro avg       0.97      0.86      0.91    199365
    weighted avg       1.00      1.00      1.00    199365
```    
Imbalanced datasets create a big problem. Like in our case if the minority class is just 0.17% of the majority. The model might resort to simplifying it as just "one" class and inturn get great accuracy. In other words the model constantly guesses that there is no fraud and get away with it. To prevent this we need to artificial increase the importance of the minority. This is done predominantly by 2 methods:

1. **UnderSampling** - We reduce the number of entries in the majority class by deleting them randomly, to make the overall ratio better. 
2. **OverSampling** - We increase the entried in the minority by duplicating them without replacement. This also makes the overall ratio better.


Let's try a method called [**undersampling**](https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/).

## Under-Sampling


```python
from imblearn.under_sampling import NearMiss
ns = NearMiss(0.8)
X_train_ns, y_train_ns = ns.fit_resample(X_train, y_train)
```

```python
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_ns)))
```

    The number of classes before fit Counter({0: 85294, 1: 148})
    The number of classes after fit Counter({0: 185, 1: 148})
```

Now we have equalized the classes, lets see if our model performs any better.
    
```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train_ns, y_train_ns)
```


```python
prediction = clf.predict(X_test)
print(confusion_matrix(y_test, prediction))
print(accuracy_score(y_test, prediction))
print(classification_report(y_test, prediction))
```

    [[177877  21144]
     [    26    318]]
    0.8938128558172197


                  precision    recall  f1-score   support
    
               0       1.00      0.89      0.94    199021
               1       0.01      0.92      0.03       344
    
        accuracy                           0.89    199365
       macro avg       0.51      0.91      0.49    199365
    weighted avg       1.00      0.89      0.94    199365
    

We have improved our recall, but our precision is dismal. this is a true disaster. Let's try [**over-sampling**](https://analyticsindiamag.com/handling-imbalanced-datasets-a-guide-with-hands-on-implementation/).
    

## Over-Sampling



```python
from imblearn.over_sampling import RandomOverSampler
os = RandomOverSampler(0.5)
X_train_os, y_train_os = os.fit_resample(X_train, y_train)
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_os)))
```

    The number of classes before fit Counter({0: 85294, 1: 148})
    The number of classes after fit Counter({0: 85294, 1: 42647})

```python
os = RandomOverSampler(0.5)
X_train_os, y_train_os = os.fit_resample(X_train, y_train)
```

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train_os, y_train_os)
```

```python
prediction = clf.predict(X_test)
print(confusion_matrix(y_test, prediction))
print(accuracy_score(y_test, prediction))
print(classification_report(y_test, prediction))
```

    [[199004     17]
     [    81    263]]
    0.9995084392947609
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00    199021
               1       0.94      0.76      0.84       344
    
        accuracy                           1.00    199365
       macro avg       0.97      0.88      0.92    199365
    weighted avg       1.00      1.00      1.00    199365


That looks a little better! a good precision with a decent recall. Lets try a [**SMOTETomek model**](https://imbalanced-learn.org/dev/references/generated/imblearn.combine.SMOTETomek.html).
    

## SMOTETomek


```python
from imblearn.combine import SMOTETomek
```


```python
os = SMOTETomek(0.5)
X_train_os, y_train_os = os.fit_resample(X_train, y_train)
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_os)))
```

    The number of classes before fit Counter({0: 85294, 1: 148})
    The number of classes after fit Counter({0: 84204, 1: 41557})


```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train_os, y_train_os)
```



```python
prediction = clf.predict(X_test)
print(confusion_matrix(y_test, prediction))
print(accuracy_score(y_test, prediction))
print(classification_report(y_test, prediction))
```

    [[198975     46]
     [    62    282]]
    0.9994582800391242
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00    199021
               1       0.86      0.82      0.84       344
    
        accuracy                           1.00    199365
       macro avg       0.93      0.91      0.92    199365
    weighted avg       1.00      1.00      1.00    199365
    
Wow that's improved the recall a lot but the precision has dropped. I'm guessing that's because of the part undersampling that the SMOTETomek model does. Why not try just the [**SMOTE model**](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html).

## SMOTE


```python
from imblearn.over_sampling import SMOTE
```


```python
sm = SMOTE()
X_train_sm, y_train_sm = os.fit_resample(X_train, y_train)
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_sm)))
```

    The number of classes before fit Counter({0: 85294, 1: 148})
    The number of classes after fit Counter({0: 84230, 1: 41583})
    

```python
from sklearn.ensemble import RandomForestClassifier
clfsm = RandomForestClassifier()
clfsm.fit(X_train_sm, y_train_sm)
```
    [[198976     45]
     [    55    289]]
    0.9994984074436335
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00    199021
               1       0.87      0.84      0.85       344
    
        accuracy                           1.00    199365
       macro avg       0.93      0.92      0.93    199365
    weighted avg       1.00      1.00      1.00    199365
    
Yes! we got slightly better results on both the precision and recall metrics. There is just one more model i wanted to try out, thats the [**Extra-Trees Classifier**](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html).   

## Extra-Trees Classifier


```python
from sklearn.ensemble import ExtraTreesClassifier
clfsm = ExtraTreesClassifier()
clfsm.fit(X_train_sm, y_train_sm)
```

```python
prediction = clfsm.predict(X_test)
print(confusion_matrix(y_test, prediction))
print(accuracy_score(y_test, prediction))
print(classification_report(y_test, prediction))
```

    [[198977     44]
     [    55    289]]
    0.9995034233691972
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00    199021
               1       0.87      0.84      0.85       344
    
        accuracy                           1.00    199365
       macro avg       0.93      0.92      0.93    199365
    weighted avg       1.00      1.00      1.00    199365
    
    
The results of the ExtraTrees and SMOTE model look quite simliar.. thats interesting

## Conclusions

We have tried to solve our problem of data imbalance using multiple approaches. The best model that we could produce was between the extra-trees and the SMOTE model. Further attempts could be made with other models.

To check out the complete code for the project [**click here**](https://github.com/realnihal/Credit-Card-Fraud-Problem).

And with that Peace out!
