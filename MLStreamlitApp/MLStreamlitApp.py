# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz

# Creates title
st.title("Applying Supervised Machine Learning Processes to Various Datasets")
# Tells the user initial instructions
st.write("### Step 1) Select the Dataset You Want to Analyze")

# Prompts user to select a dataframe to apply Machine Learning analysis on
dataset = st.selectbox("Select a dataset:", ["Titanic", "Car Crashes", "Exercise", "Upload My Own"])

# If the Titanic box is selected, the titanic dataset is loaded in
if dataset == "Titanic":
    df = sns.load_dataset('titanic')
# If the car crashes box is selected, the car crashes dataset is loaded in
elif dataset == "Car Crashes":
    df = sns.load_dataset('car_crashes')
# If the exercise box is selected, the exercise dataset is loaded in
elif dataset == "Exercise":
    df = sns.load_dataset('exercise')
# If the Upload My Own box is selected, the app prompts the user to upload their own dataset
elif dataset == "Upload My Own":
    # Allows user to upload their own CSV file
    uploaded_file = st.file_uploader("Upload a CSV", type = "csv")
    # Ensures a file is uploaded before it runs
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    #If no file is uploaded, there is no dataset
    else:
        df = None

# Ensures that a dataset exists before any analysis is done to prevent erros
if df is not None:
    # Drops all rows that have missing values
    df = df.dropna()
    # Displays a title for the dataset
    st.write("#### Dataset:")
    # Displays a dataframe of the data
    st.dataframe(df)
    # Gives instructions to the user asking which model they want to run
    st.write("### Step 2) Select the Type of Model You Want to Look At")
    # Prompts user to select the type of model they want to run
    modeltype = st.radio("Pick one:", ["Logistic Regression", "Decision Tree"])
    # Gives the user instructions to adjust the model settings
    st.write("### Step 3) Adjust Model Settings")
    # Warns the user to only select a target variable that is discrete and categorical so the model does not give an error
    st.write("#### Disclaimer: Make sure that the target variable is a discrete, categorical variable")
    # Prompts the user to select the target variable
    target = st.selectbox("Select target variable:", df.columns)
    # Defines which variables are feature variables
    features = [col for col in df.columns if col != target]
    # Assigns the feature variables to X to be used in model
    X = df[features]
    # Sets the feature variable to y to be used in model
    y = df[target]
    # Converts feature variables to dummy variables
    X = pd.get_dummies(X, drop_first=True)
    # What the model does if the user wants to run a logistic regression model
    if modeltype == "Logistic Regression":
        # Prompts user to select test size for model
        usertestsize = st.slider("What should the test size be?", 0.01, 1.00, step = 0.01)
        # Prompts user to select the random state for model
        userrandomstate = st.slider("What should the random state be?", 0, 100)
        # Split dataset into training and testing subsets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = usertestsize, random_state = userrandomstate)
        # Initialize and train logistic regression model
        model = LogisticRegression()
        # Fits the model according to training data
        model.fit(X_train, y_train)
        # Predicts based on test data
        y_pred = model.predict(X_test)
        # Calculate accuracy score of the model
        accuracy = accuracy_score(y_test, y_pred)
        # Displays a title for accuracy score
        st.write("#### Accuracy Score:")
        # Displays the accuracy score
        st.write(accuracy)
        # Calculates classification metrics
        classificationreport = classification_report(y_test, y_pred, output_dict=True)
        # Puts the classification metrics into a dataframe report
        classificationreport_df = pd.DataFrame(classificationreport).transpose()
        # Displays title for classification report
        st.write("#### Classification Report:")
        # Displays the classification report dataframe
        st.dataframe(classificationreport_df)
        # Computes confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        # Initializes plot for confusion matrix
        fig, ax = plt.subplots()
        # Edits plot for confusion matrix, making it a heat map with blue colors
        sns.heatmap(cm, annot=True, cmap="Blues", ax=ax)
        # Creates x- and y-axis labels for confusion matrix graph
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        # Creates a title for the confusion matrix
        st.write("#### Confusion Matrix:")
        # Plots the confusion matrix graph
        st.pyplot(fig)
    # What the model does if the user wants to run a Decision Tree model
    elif modeltype == "Decision Tree":
        # Prompts user to select test size for model
        usertestsize = st.slider("What should the test size be?", 0.01, 1.00, step = 0.01)
        # Prompts user to select the random state for model
        userrandomstate = st.slider("What should the random state be?", 0, 100)
        # Prompts user to select the max tree depth for the model
        usermaxdepth = st.slider("What should the decision tree max depth be?", 1, 10)
        # Split dataset into training and testing subsets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = usertestsize, random_state = userrandomstate)
        # Initializes decision tree based on user model input 
        model = DecisionTreeClassifier(random_state = userrandomstate, max_depth= usermaxdepth)
        # Fits the decision tree model to the training data
        model.fit(X_train, y_train)
        # Predicts based on test data
        y_pred = model.predict(X_test)
        # Calculates accuracy score
        accuracy = accuracy_score(y_test, y_pred)
        # Displays title for accuracy score
        st.write("#### Accuracy Score:")
        # Displays accuracy score
        st.write(accuracy)
        # Calculates classification report
        classificationreport = classification_report(y_test, y_pred, output_dict=True)
        # Converts classification report to a dataframe
        classificationreport_df = pd.DataFrame(classificationreport).transpose()
        # Displays title for classification report
        st.write("#### Classification Report:")
        # Displays the classification report dataframe
        st.dataframe(classificationreport_df)
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        # Initializes plot for confusion matrix
        fig, ax = plt.subplots()
        # Edits plot for confusion matrix, making it a heat map with red colors
        sns.heatmap(cm, annot=True, cmap="Reds", ax=ax)
        # Creates x- and y-axis labels for confusion matrix graph
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        # Creates a title for the confusion matrix
        st.write("#### Confusion Matrix:")
        # Plots the confusion matrix graph
        st.pyplot(fig)
        # Creates a title for the decision tree 
        st.write("#### Decision Tree Graph:")
        # Generates the model and chart for the decision tree
        dot_data = tree.export_graphviz(model, # actual model
            feature_names = X_train.columns, # columns of our features
            filled= True) #gini index, values per class
        # Displays the decision tree graph
        st.graphviz_chart(dot_data)