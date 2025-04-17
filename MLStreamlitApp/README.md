# Applying Supervised Machine Learning Processes to Various Datasets!📊
## Project Overview👀
* The goal of this project is for the user to explore supervised machine learning models and apply them to real life data
* Supervised machine learning is when a model is trained on inputs to predict the outputs of a target variable
* Machine learning models allow us to make predictions, and different metrics and graphs can show us how accurate these predictions are
* In the app, the user can use preset datasets or upload their own dataset and then explore with how different inputs affect the model and its predictions

## Instructions🧭
* To run the app, you must first click on the file named "MLStreamlitApp.py" within the "MLStreamlitApp" folder
  - Additionally, you can access the app at the following link: https://mccarthy-data-science-portfolio-exploringml.streamlit.app/
* From there, click the logo labeled "Download raw file" in the top right corner
* When it downloads, click on the file and the code should pop up within Visual Studio Code or Google CoLab
* Make sure to download all of the required libraries and functions, listed below and in the "requirements.txt" channel
  1. graphviz==0.20.1
  2. matplotlib==3.10.1
  3. numpy==2.2.4
  4. pandas==2.2.3
  5. scikit_learn==1.6.1
  6. seaborn==0.13.2
  7. streamlit==1.37.1
* From there, follow the instructions written out in the app, starting with Step 1

## App Features🏷️
* The app allows the user some flexibility to explore as they so choose
  - Users can select from 3 provided sample datasets or upload their own CSV file
* Users can choose to either run a linear regression, logistic regression, or decision tree model on the data
  - Linear regression models are used to predict a numeric variable when a roughly linear relationship exists with other variables
  - Logistic regression models are used to predict a binary, categorical variable that's outcome is influenced by multiple features
  - Decision tree models make predictions based on a series of Yes/No questions and mimic human decision-making
* Users can also use sliders to manually change the models hyperparameters, including the target variable, test size, random state, and decision tree max depth
  - Users can explore how each of these affects the model's output
 
## Visuals📸
<img width="530" alt="image" src="https://github.com/user-attachments/assets/2c0fed3e-ddc9-459b-8d5a-588e8c01a2c2" />

An example of a dataset that the user could input, as shown in the app

<img width="560" alt="image" src="https://github.com/user-attachments/assets/875d7396-6810-4e08-8e2c-0680f3f57ef7" />

An example of the hyperparameters that the user can control using sliders

<img width="277" alt="image" src="https://github.com/user-attachments/assets/1a7ef56f-de97-47c5-901a-4a6b2194dc21" />

An example of the classification report output, giving various metrics on the model and its accuracy

<img width="535" alt="image" src="https://github.com/user-attachments/assets/a065ba37-f1f8-486e-90d6-bc806bf53d24" />

An example of a confusion matrix output

<img width="374" alt="image" src="https://github.com/user-attachments/assets/de407b06-c4f4-4554-83c4-a1c5d8ed1560" />

An example of a decision tree graph


## Sources Used🔗
  - https://docs.streamlit.io/develop/quick-reference/cheat-sheet
  - https://seaborn.pydata.org/generated/seaborn.load_dataset.html
  - https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf 
  
