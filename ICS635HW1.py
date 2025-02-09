# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:51:09 2025

@author: 808al
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

#Data Preprocessing

#load data from breast cancer database and split the inputs and outputs
input_data, output_data = load_breast_cancer(return_X_y=True)

#split the data into a training set and a testing set, with 20% in testing
x_train, x_test, y_train, y_test = train_test_split(
    input_data, output_data, test_size = 0.2, random_state = 0)

#scale the features for KNN
x_train_scaler = preprocessing.StandardScaler().fit(x_train)
x_train_scaled = x_train_scaler.transform(x_train)

x_test_scaler = preprocessing.StandardScaler().fit(x_test)
x_test_scaled = x_test_scaler.transform(x_test)

#Model Training

#KNN
knn_model = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn_model.fit(x_train_scaled, y_train)

#Decision Tree
decision_tree_model = tree.DecisionTreeClassifier()
decision_tree_model.fit(x_train, y_train)

#Random Forest
random_forest_model = RandomForestClassifier(max_depth=3, 
                                             n_estimators=100, 
                                             max_features="sqrt", 
                                             bootstrap=True)
random_forest_model.fit(x_train, y_train)

#Model Testing

knn_prediction = knn_model.predict(x_test_scaled)
decision_tree_prediction = decision_tree_model.predict(x_test)
random_forest_prediction = random_forest_model.predict(x_test)

#Model Evaluation

def EvaluateModel(prediction):
    return [
        round(accuracy_score(y_test, prediction), 4), 
        round(precision_score(y_test, prediction), 4),
        round(recall_score(y_test, prediction), 4),
        round(f1_score(y_test, prediction), 4)
        ]

knn_metrics = EvaluateModel(knn_prediction)
decision_tree_metrics = EvaluateModel(decision_tree_prediction)
random_forest_metrics = EvaluateModel(random_forest_prediction)

#Table
table_data = [["KNN"] + knn_metrics, 
              ["Decision Tree"] + decision_tree_metrics,
              ["Random Forest"] + random_forest_metrics]

figure, axes = plt.subplots(7, 1, figsize=(6, 36))

axes[0].axis("tight")
axes[0].axis("off")
axes[0].set_title("Model Evaluation")
table = axes[0].table(cellText=table_data, colLabels=["Model", "Accuracy", "Precision", "Recall", "F1 Score"], cellLoc="center", loc="center")

#Confusion Matrices
knn_confusion = confusion_matrix(y_test, knn_prediction)
decision_tree_confusion = confusion_matrix(y_test, decision_tree_prediction)
random_forest_confusion = confusion_matrix(y_test, random_forest_prediction)

axes[1].set_title("KNN Confusion Matrix")
ConfusionMatrixDisplay(knn_confusion, display_labels=["Benign", "Malignant"]).plot(ax=axes[1])

axes[2].set_title("Decision Tree Confusion Matrix")
ConfusionMatrixDisplay(decision_tree_confusion, display_labels=["Benign", "Malignant"]).plot(ax=axes[2])

axes[3].set_title("Random Forest Confusion Matrix")
ConfusionMatrixDisplay(random_forest_confusion, display_labels=["Benign", "Malignant"]).plot(ax=axes[3])

#Ablation Studies

#KNN Ablation
def KNNAblation(num_neighbors):
    model = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors, weights='uniform')
    model.fit(x_train_scaled, y_train)
    model_prediction = model.predict(x_test_scaled)
    return EvaluateModel(model_prediction)

knn_ablation_data = [[3] + KNNAblation(3),
                     [5] + knn_metrics,
                     [7] + KNNAblation(7),
                     [9] + KNNAblation(9)]

axes[4].axis("tight")
axes[4].axis("off")
axes[4].set_title("KNN Ablation")
table = axes[4].table(cellText=knn_ablation_data, colLabels=["K Value", "Accuracy", "Precision", "Recall", "F1 Score"], cellLoc="center", loc="center")

#Decision Tree Ablation
def DecisionTreeAblation(depth):
    model = tree.DecisionTreeClassifier(max_depth=depth)
    model.fit(x_train, y_train)
    model_prediction = model.predict(x_test)
    return EvaluateModel(model_prediction)

decision_tree_ablation_data = [[3] + DecisionTreeAblation(3),
                               [5] + DecisionTreeAblation(5),
                               [7] + DecisionTreeAblation(7),
                               [9] + DecisionTreeAblation(9)]

axes[5].axis("tight")
axes[5].axis("off")
axes[5].set_title("Decision Tree Ablation")
table = axes[5].table(cellText=decision_tree_ablation_data, colLabels=["Max Depth", "Accuracy", "Precision", "Recall", "F1 Score"], cellLoc="center", loc="center")

#Random Forest Ablation
def RandomForestAblation(depth):
    model = RandomForestClassifier(max_depth=depth, 
                                   n_estimators=100, 
                                   max_features="sqrt", 
                                   bootstrap=True)
    model.fit(x_train, y_train)
    model_prediction = model.predict(x_test)
    return EvaluateModel(model_prediction)

random_forest_ablation_data = [[3] + RandomForestAblation(3),
                               [5] + RandomForestAblation(5),
                               [7] + RandomForestAblation(7),
                               [9] + RandomForestAblation(9)]

axes[6].axis("tight")
axes[6].axis("off")
axes[6].set_title("Random Forest Ablation")
table = axes[6].table(cellText=random_forest_ablation_data, colLabels=["Max Depth", "Accuracy", "Precision", "Recall", "F1 Score"], cellLoc="center", loc="center")
