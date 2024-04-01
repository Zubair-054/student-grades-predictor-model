import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

pd.options.mode.chained_assignment = None


# Replace the path with the location of your Excel file
excel_file = pd.ExcelFile('DM_Dataset.xlsx')

df1 = excel_file.parse('D1')
df2 = excel_file.parse('D2')
df3 = excel_file.parse('D3')
df4 = excel_file.parse('D4')
df5 = excel_file.parse('D5')
df6 = excel_file.parse('D6')
df7 = excel_file.parse('D7')

def weightAdder(df):
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(index=2, axis=0)
    # Loop through columns and scale if needed
    for col in df.columns:
        if col.startswith('As:') or col.startswith('Qz:'):
            weight = df.loc[0, col]
            totalMarks = df.loc[1, col]
            scaling_factor = weight/totalMarks
            df.loc[2:, col] = round(df.loc[2:, col] * scaling_factor,2)
    df = df.drop(index=[0,1])
    df = df.reset_index(drop=True)
    return df
def getmaxAsnQz5(df):
    as_columns = [col for col in df.columns if (col.startswith('As:'))]
    as_df = df[as_columns]
    top5_max = as_df.apply(lambda x: x.nlargest(5).values, axis=1)
    for i in range(len(as_df)):
        for j in range(len(as_df.iloc[i,:])):
            as1 = as_df.iloc[i,:][j]
            if as1 not in top5_max[i]:
                as_df.iloc[i, :][j] = 0
    cols = as_df.columns
    df[cols] = as_df[cols]
    df = getmaxQz5(df)
    return df
def getmaxQz5(df):
    as_columns = [col for col in df.columns if (col.startswith('Qz:'))]
    as_df = df[as_columns]
    top5_max = as_df.apply(lambda x: x.nlargest(5).values, axis=1)
    for i in range(len(as_df)):
        for j in range(len(as_df.iloc[i,:])):
            as1 = as_df.iloc[i,:][j]
            if as1 not in top5_max[i]:
                as_df.iloc[i, :][j] = 0
    cols = as_df.columns
    df[cols] = as_df[cols]
    return df

def shiftTo5columns(df):
    as_columns = [col for col in df.columns if (col.startswith('As:'))]
    qz_columns = [col for col in df.columns if (col.startswith('Qz:'))]
    qz_df = df[qz_columns]
    as_df = df[as_columns]
    for index, row in as_df.iterrows():
        row_copy = row.copy()
        row_copy = row_copy.sort_values(kind='mergesort', key=lambda x: (x == 0)).values
        row_copy = np.concatenate([row[row != 0], row_copy[row_copy == 0]])
        as_df.loc[index] = row_copy
    for index, row in qz_df.iterrows():
        row = row.sort_values(kind='mergesort', key=lambda x: (x == 0)).values
        row = np.concatenate([row[row != 0], row[row == 0]])
        qz_df.loc[index] = row
    cols = ['As:1','As:2','As:3','As:4','As:5','As','Qz:1','Qz:2','Qz:3','Qz:4','Qz:5','Qz','S-I','S-II','Grade']
    drop_cols = [col for col in df.columns if col not in cols]
    drop_cols1 = [col for col in as_df.columns if col not in cols]
    drop_cols2 = [col for col in qz_df.columns if col not in cols]
    df = df.drop(drop_cols, axis=1)
    as_df = as_df.drop(drop_cols1,axis=1)
    qz_df = qz_df.drop(drop_cols2,axis=1)

    cols_name = as_df.columns
    qz_cols_name = qz_df.columns
    df[cols_name] = as_df[cols_name]
    df[qz_cols_name] = qz_df[qz_cols_name]

    return df
#Adding the Weight to the Assignments & Quizes
df1 = weightAdder(df1)
df2 = weightAdder(df2)
df3 = weightAdder(df3)
df4 = weightAdder(df4)
df5 = weightAdder(df5)
df6 = weightAdder(df6)
df7 = weightAdder(df7)

# filling the null values
df1 = df1.fillna(0)
df2 = df2.fillna(0)
df3 = df3.fillna(0)
df4 = df4.fillna(0)
df5 = df5.fillna(0)
df6 = df6.fillna(0)
df7 = df7.fillna(0)

ddf1 = shiftTo5columns(df1)
ddf2 = shiftTo5columns(df2)
ddf3 = shiftTo5columns(df3)
ddf4 = shiftTo5columns(df4)
ddf5 = shiftTo5columns(df5)
ddf6 = shiftTo5columns(df6)
ddf7 = shiftTo5columns(df7)

df1 = getmaxAsnQz5(df1)
df1 = shiftTo5columns(df1)

df2 = getmaxAsnQz5(df2)
df2 = shiftTo5columns(df2)

df3 = getmaxAsnQz5(df3)
df3 = shiftTo5columns(df3)

df4 = getmaxAsnQz5(df4)
df4 = shiftTo5columns(df4)

df5 = getmaxAsnQz5(df5)
df5 = shiftTo5columns(df5)

df6 = getmaxAsnQz5(df6)
df6 = shiftTo5columns(df6)

df7 = getmaxAsnQz5(df7)
df7 = shiftTo5columns(df7)


# All Data Values are not having same column numbers
# print(df1.shape)
# print(df2.shape)
# print(df3.shape)
# print(df4.shape)
# print(df5.shape)
# print(df6.shape)
# print(df7.shape)

dff = pd.concat([ddf1,ddf2,ddf3,ddf4,ddf5,ddf6,ddf7])
dff = dff.reset_index(drop=True)

df = pd.concat([df1,df2,df3,df4,df5,df6,df7])
df = df.reset_index(drop=True)

# Basic Stats
# print(df.describe())

#Removing Outliers

# Q1 = df.quantile(0.25)
# Q3 = df.quantile(0.75)
# IQR = Q3 - Q1
#
# # Remove outliers
# df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]




def drawCorrelationMatrix(df):
    # Defining the Correlation Matrix
    corr_matrix = df.corr()
    print("Correlation Matrix: \n",corr_matrix)
    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu")
    plt.title("Correlation Matrix")
    plt.show()


drawCorrelationMatrix(df)


def drawHistrogram(df):
    # Iterate over each column in the dataframe
    for col in df.columns:
        if col != 'Grade':
            # Draw histogram
            sns.histplot(x=col, data=df, kde=True)
            plt.show()


drawHistrogram(df)


def drawBoxPlot(df):
    # Iterate over each column in the dataframe
    for col in df.columns:
        if col != 'Grade':
            # Draw box plot
            sns.boxplot(x='Grade', y=col, data=df)
            plt.show()


# drawBoxPlot(df)


# Separate out the features (i.e., columns other than target)

# def predict_class(dt,knn,nb,input_values):
#
#     # Scale the input values using the same StandardScaler object
#     input_values = scaler.transform([input_values])
#
#     # Predict the class label for the input values using each classifier
#     dt_pred = dt.predict(input_values)
#     knn_pred = knn.predict(input_values)
#     nb_pred = nb.predict(input_values)
#
#     # Perform majority voting to classify the input values as Pass or Fail
#     votes = [dt_pred[0], knn_pred[0], nb_pred[0]]
#     pass_votes = votes.count("Pass")
#     fail_votes = votes.count("Fail")
#     if pass_votes > fail_votes:
#         return "Pass"
#     else:
#         return "Fail"

# def get_input():
#     # Ask user whether they want to predict before or after S-II
#     features = ""
#     prediction_value = ""
#     while True:
#         prediction_value = ""
#         prediction_type = input(
#             "Do you want to predict grades before or after S-II? Enter 'B' for before and 'A' for after: ")
#         if prediction_type.upper() == 'B':
#             features = ['As:1', 'As:2', 'As:3', 'As:4', 'Qz:1', 'Qz:2', 'Qz:3',
#                         'Qz:4', 'S-I']
#             prediction_value = 'B'
#             break
#         elif prediction_type.upper() == 'A':
#             features = ['As:1', 'As:2', 'As:3', 'As:4', 'As:5', 'Qz:1',
#                         'Qz:2', 'Qz:3', 'Qz:4', 'Qz:5', 'S-I', 'S-II', 'Total']
#             prediction_value = 'A'
#             break
#         else:
#             print("Invalid input. Please enter 'B' or 'A'.")
#
#     # Get input values for the chosen features
#     while True:
#         try:
#             assignment_scores = [float(input("Enter score for {}: ".format(feature))) for feature in features if
#                                  'Assignment' in feature]
#             quiz_scores = [float(input("Enter score for {}: ".format(feature))) for feature in features if
#                            'Quiz' in feature]
#             S_i_score = float(input("Enter score for S-I: "))
#             if max(assignment_scores) > 3 or max(quiz_scores) > 3 or S_i_score > 15:
#                 raise ValueError("Scores cannot exceed 3 for assignments/quizzes and 15 for S-I.")
#             break
#         except ValueError as e:
#             print("Invalid input:", e)
#             continue
#
#     # Combine the input values into a single list
#     input_values = assignment_scores + quiz_scores + [S_i_score]
#     if prediction_type == "B":
#         X_1 = df[['As:1', 'As:2', 'As:3', 'As:4', 'Qz:1', 'Qz:2', 'Qz:3', 'Qz:4',
#                 'S-I']]
#         y_1 = df['Grade']
#     if prediction_type == "S-II":
#         X_1 = df[['As:1', 'As:2', 'As:3', 'As:4','As:5', 'Qz:1', 'Qz:2', 'Qz:3', 'Qz:4','Qz:5',
#                 'S-I','S-II']]
#         y_1 = df['Grade']
#
#     # Return the input values
#     return input_values,X_1,y_1
#
    # # input_values,X,y = get_input()

# df = df[['As:1', 'As:2', 'As:3', 'As:4','As:5', 'Qz:1', 'Qz:2', 'Qz:3', 'Qz:4','Qz:5','S-I','S-II','Grade']]
df = df[['As:1', 'As:2', 'As:3', 'As:4','As:5', 'Qz:1', 'Qz:2', 'Qz:3', 'Qz:4','Qz:5','S-I','S-II','Grade']]
# df = dff[['As:1', 'As:2', 'As:3', 'As:4', 'Qz:1', 'Qz:2', 'Qz:3', 'Qz:4','S-I','Grade']]
X = df.drop('Grade',axis=1)
y = df['Grade']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 1)


def getBestK():
    k_range = range(1, 25)
    train_accuracy = []
    test_accuracy = []
    # List to store cross-validation scores
    cv_scores = []
    # 10-fold cross-validation
    for i, k in enumerate(k_range):
        # k from 1 to 25(exclude)
        knn = KNeighborsClassifier(n_neighbors=k)
        # Fit with knn
        knn.fit(x_train, y_train)
        # train accuracy
        train_accuracy.append(knn.score(x_train, y_train))
        # test accuracy
        test_accuracy.append(knn.score(x_test, y_test))
    # Plot cross-validation scores vs K values
    # Plot
    plt.figure(figsize=[13, 8])
    plt.plot(k_range, test_accuracy, label='Testing Accuracy')
    plt.plot(k_range, train_accuracy, label='Training Accuracy')
    plt.legend()
    plt.title('-value VS Accuracy')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.xticks(k_range)
    plt.savefig('graph.png')
    plt.show()
    print(
        "Best accuracy is {} with K = {}".format(np.max(test_accuracy), 1 + test_accuracy.index(np.max(test_accuracy))))
    return 1 + test_accuracy.index(np.max(test_accuracy))



# Define range of K values to test


K = getBestK()

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize classifiers
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier(n_neighbors=K)
nb = GaussianNB()


# Initialize Leave-One-Out cross-validator
loo = LeaveOneOut()

dt_accuracy_list = []
knn_accuracy_list = []
nb_accuracy_list = []

dt_predictions = []
knn_predictions = []
nb_predictions = []

positive_label = 'Pass'  # Set the positive label value

for train_index, test_index in loo.split(X):
    # Split the data into training and testing sets for the current iteration
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the Decision Tree classifier and compute accuracy
    dt.fit(X_train, y_train)
    dt_predictions.append(dt.predict(X_test))
    dt_accuracy = dt.score(X_test, y_test)
    dt_accuracy_list.append(dt_accuracy)

    # Fit the KNN classifier and compute accuracy
    knn.fit(X_train, y_train)
    knn_predictions.append(knn.predict(X_test))
    knn_accuracy = knn.score(X_test, y_test)
    knn_accuracy_list.append(knn_accuracy)

    # Fit the Naive Bayes classifier and compute accuracy
    nb.fit(X_train, y_train)
    nb_predictions.append(nb.predict(X_test))
    nb_accuracy = nb.score(X_test, y_test)
    nb_accuracy_list.append(nb_accuracy)

# Compute mean accuracy for Decision Tree and KNN
dt_mean_accuracy = np.mean(dt_accuracy_list)
knn_mean_accuracy = np.mean(knn_accuracy_list)
nb_mean_accuracy = np.mean(nb_accuracy_list)

print("Decision Tree Mean accuracy after Leave One Out Validation: {:.2f}%".format(dt_mean_accuracy * 100))
print("KNN Mean accuracy after Leave One Out Validation: {:.2f}%".format(knn_mean_accuracy * 100))
print("Naive Bayes Mean accuracy after Leave One Out Validation: {:.2f}%".format(nb_mean_accuracy * 100))

# Compute confusion matrix for each classifier
dt_confusion_matrix = confusion_matrix(y, np.concatenate(dt_predictions), labels=['Fail', 'Pass'])
knn_confusion_matrix = confusion_matrix(y, np.concatenate(knn_predictions), labels=['Fail', 'Pass'])
nb_confusion_matrix = confusion_matrix(y, np.concatenate(nb_predictions), labels=['Fail', 'Pass'])

# Calculate evaluation metrics
dt_specificity = dt_confusion_matrix[0, 0] / (dt_confusion_matrix[0, 0] + dt_confusion_matrix[0, 1])
dt_recall = dt_confusion_matrix[1, 1] / (dt_confusion_matrix[1, 0] + dt_confusion_matrix[1, 1])
dt_precision = precision_score(y, np.concatenate(dt_predictions), pos_label=positive_label)
dt_f1_score = f1_score(y, np.concatenate(dt_predictions), pos_label=positive_label)

knn_specificity = knn_confusion_matrix[0, 0] / (knn_confusion_matrix[0, 0] + knn_confusion_matrix[0, 1])
knn_recall = knn_confusion_matrix[1, 1] / (knn_confusion_matrix[1, 0] + knn_confusion_matrix[1, 1])
knn_precision = precision_score(y, np.concatenate(knn_predictions), pos_label=positive_label)
knn_f1_score = f1_score(y, np.concatenate(knn_predictions), pos_label=positive_label)

nb_specificity = nb_confusion_matrix[0, 0] / (nb_confusion_matrix[0, 0] + nb_confusion_matrix[0, 1])
nb_recall = nb_confusion_matrix[1, 1] / (nb_confusion_matrix[1, 0] + nb_confusion_matrix[1, 1])
nb_precision = precision_score(y, np.concatenate(nb_predictions), pos_label=positive_label)
nb_f1_score = f1_score(y, np.concatenate(nb_predictions), pos_label=positive_label)

print("Decision Tree Metrics:")
print("Specificity: {:.2f}".format(dt_specificity))
print("Recall: {:.2f}".format(dt_recall))
print("Precision: {:.2f}".format(dt_precision))
print("F1 Score: {:.2f}".format(dt_f1_score))
print("")

print("KNN Metrics:")
print("Specificity: {:.2f}".format(knn_specificity))
print("Recall: {:.2f}".format(knn_recall))
print("Precision: {:.2f}".format(knn_precision))
print("F1 Score: {:.2f}".format(knn_f1_score))
print("")

print("Naive Bayes Metrics:")
print("Specificity: {:.2f}".format(nb_specificity))
print("Recall: {:.2f}".format(nb_recall))
print("Precision: {:.2f}".format(nb_precision))
print("F1 Score: {:.2f}".format(nb_f1_score))




# Parallel Coordinate Plot and Pairwise Plot
from pandas.plotting import andrews_curves
plt.figure(figsize=(15,10))
andrews_curves(df.drop(['As:1','As:2','As:3','As:4','Qz:1','Qz:2','Qz:3','Qz:4'],axis=1), "Grade"
                     ,color=["green","red"])
plt.title('Parallel Coordinates Plot', fontsize=20, fontweight='bold')
plt.xlabel('Grade', fontsize=15)
plt.ylabel('Marks', fontsize=15)
plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
plt.show()

plt.figure()
sns.pairplot(df.drop(['As:1','As:2','As:3','As:4','Qz:1','Qz:2','Qz:3','Qz:4'],axis=1), hue = "Grade", size=2, markers=[ "s", "D"])
plt.show()


