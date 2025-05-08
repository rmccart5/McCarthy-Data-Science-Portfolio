# Applying Unsupervised Machine Learning Processes to Datasets!🎰
## Project Overview👀
* The goal of this project is for the user to explore unsupervised machine learning models and apply them to real life data
* Unsupervised machine learning is when a model learns patterns from unlabeled data, meaning the data doesn't have predefined labels or categories
* Machine learning models allow us to make predictions, and different metrics and graphs can show us how accurate these predictions are
* In the app, the user can use preset datasets or upload their own dataset and then explore with how different inputs affect the model and its predictions

## Instructions🧭
* To run the app, you must first click on the file named "MLUnsupervisedApp.py" within the "MLUnsupervisedApp" folder
  - Additionally, you can access the app at the following link: https://mccarthy-data-science-portfolio-mlunsupervisedapp.streamlit.app/
* From there, click the logo labeled "Download raw file" in the top right corner
* When it downloads, click on the file and the code should pop up within Visual Studio Code or Google CoLab
* Make sure to download all of the required libraries and functions, listed below and in the "requirements.txt" channel within the "MLUnsupervisedApp" folder
  1. matplotlib==3.10.3
  2. numpy==2.2.5
  3. pandas==2.2.3
  4. scikit_learn==1.6.1
  5. scipy==1.15.3
  6. seaborn==0.13.2
  7. streamlit==1.37.1

* From there, follow the instructions written out in the app, starting with Step 1

## App Features🏷️
* The app allows the user some flexibility to explore as they so choose
  - Users can select from 3 provided sample datasets or upload their own CSV file
* Users can choose to either run a K-Means Clustering, Hierarchical Clustering, or Principal Component Analysis model on the data
  - K-Means Clustering is an algorithm that groups similar data points into clusters based on their features
  - Hierarchical Clustering starts with each data point as its own cluster and then iteratively merges the most similar clusters until a single, large cluster remains
  - Principal Component Analysis reduces the number of variables in a dataset while minimizing the amount of significant information lost
* Users can also use sliders to manually change the models hyperparameters, including the target variable, feature variables, and number of clusters
  - Users can explore how each of these affects the model's output
 
## Visuals📸
![image](https://github.com/user-attachments/assets/29693496-162b-44d2-bded-2e45ebde04d4)

An example of a datset output

![image](https://github.com/user-attachments/assets/dbaa9b1f-894c-4658-b470-580e33362bd2)

A K-Means Clustering scatterplot output

![image](https://github.com/user-attachments/assets/5349a583-fc9b-4b60-b3de-c6443c515871)

An example Silhouette Score graph for K-Means Clustering

![image](https://github.com/user-attachments/assets/9e9abf0a-2fc1-42db-9d75-d8e028308581)

An example of a dedrogram

![image](https://github.com/user-attachments/assets/42e74a59-e451-44fb-b5aa-2d5bf66d8719)

An example of a variance bar plot for Principal Component Analysis


## Sources Used🔗
  - https://docs.streamlit.io/develop/quick-reference/cheat-sheet
  - https://seaborn.pydata.org/generated/seaborn.load_dataset.html
  - https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf 
  

