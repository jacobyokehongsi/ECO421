import pydotplus
from six import StringIO
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

np.random.seed(0)

# Read the input csv file
dataset = pd.read_csv("question1.csv")

# Split the data into features and target

features = dataset.drop("Dropout", axis=1)
targets = dataset["Dropout"]

# Split the data into a training and a testing set
train_features, test_features, train_targets, test_targets = \
        train_test_split(features, targets, train_size=0.75)

tree = DecisionTreeClassifier(criterion="entropy", max_depth=10)
tree = tree.fit(train_features, train_targets)

# Predict the classes of new, unseen data
prediction = tree.predict(test_features)

# Check the accuracy
score = tree.score(test_features, test_targets)
print("The prediction accuracy is: {:0.2f}%".format(score * 100))

dot_data = StringIO()
# highest_accuracy_model = DecisionTreeClassifier(criterion="entropy", max_depth=80)
# highest_accuracy_model = highest_accuracy_model.fit(train_data, train_label)

names = list(dataset.columns)[:3]
export_graphviz(tree, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                feature_names=names, class_names=['Drop', 'Dont Drop'], max_depth=5)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('visualization.png')
