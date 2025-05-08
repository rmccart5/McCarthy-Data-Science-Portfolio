# Importing necessary libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Creates title
st.title("Applying Unsupervised Machine Learning Processes to Datasets")
# Tells the user initial instructions
st.write("### Step 1) Select the Dataset You Want to Analyze")

# Prompts user to select a dataframe to apply Unsupervised ML analysis on
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
    st.write("##### Dataset:")
    # Displays a dataframe of the data
    st.dataframe(df)
    # Gives instructions to the user asking which model they want to run
    st.write("### Step 2) Select the Type of Model You Want to Run")
    # Prompts user to select the type of model they want to run
    modeltype = st.radio("Pick one:", ["K-Means Clustering", "Hierarchical Clustering", "Principal Component Analysis"])


    # What the model does if the user wants to run a K-Means Clustering model
    if modeltype == "K-Means Clustering":
        # Creates a title for the model adjustment section
        st.write("### Step 3) Adjust Model Settings")
        # Prompts the user to select the target variable
        targets = st.selectbox("Select target variable:", df.columns)
        # Prompts the user to select the feature variables
        features = st.multiselect("Select feature variables:", df.columns)

        # Ensures that the user selects at least two feature variables so that an error does not occur
        if len(features) < 2:
            # If the user does not select at least two feature variables, a warning is displayed
            st.warning("Please select at least two feature variables.")
        # If the user selects at least two feature variables, the model runs
        else:
            # Assigns the feature variables to X to be used in the model
            X = df[features]
            # Assigns the target variable to y to be used in the model
            y = df[targets]
            # Converts feature variables to dummy variables
            X = pd.get_dummies(X, drop_first=True)
            # Imports the StandardScaler to be used on the data
            from sklearn.preprocessing import StandardScaler
            # Standardizes the numeric features so that they have a mean of 0 and a standard deviation of 1
            scaler = StandardScaler()
            # Fits the scaler to the data and transforms it
            X_std = scaler.fit_transform(X)
            # Imports the KMeans model to be used on the data
            from sklearn.cluster import KMeans
            # Prompts user to select the number of clusters
            k = st.slider("How many clusters should there be?", 2, 10, step = 1)
            # Creates a title for the analysis section
            st.write("### Step 4) Analyze the Results")
            # Fits the KMeans model to the data based on the number of clusters selected by the user
            kmeans = KMeans(n_clusters=k, random_state=42)
            # Assigns the cluster labels to the data
            clusters = kmeans.fit_predict(X_std)
            # Imports the PCA to be used on the data
            from sklearn.decomposition import PCA
            # Reduce the data to 2 dimensions for visualization using PCA
            pca = PCA(n_components=2)
            # Fits the PCA to the data and transforms it
            X_pca = pca.fit_transform(X_std)
            # Creates a title for the scatterplot
            st.write("##### Scatterplot:")
            # Creates a description of the scatterplot
            st.write("This scatterplot shows the KMeans clustering results projected onto the first two principal components.")
            # Initiates the scatterplot
            fig, ax = plt.subplots(figsize=(8, 6))
            # For each cluster, it plots the data points in that cluster with a different color
            for cluster in np.unique(clusters):
                plt.scatter(X_pca[clusters == cluster, 0], X_pca[clusters == cluster, 1], 
                        label=f"Cluster {cluster}", s=60, edgecolor='k', alpha=0.7)
            # Creates the x axis label for the scatterplot
            plt.xlabel('Principal Component 1')
            # Creates the y axis label for the scatterplot
            plt.ylabel('Principal Component 2')
            # Creates the title for the scatterplot
            plt.title('KMeans Clustering: 2D PCA Projection')
            # Creates the legend for the scatterplot
            plt.legend(loc='best')
            # Creates the grid for the scatterplot
            plt.grid(True)
            # Displays the scatterplot in the app
            st.pyplot(fig)

            # Imports the silhouette_score to be used on the data
            from sklearn.metrics import silhouette_score
            # Define the range of k values to try
            ks = range(2, 11)  # starting from 2 clusters to 10 clusters
            # Initialize lists to store WCSS
            wcss = []               
            # Initialize lists to store silhouette scores
            silhouette_scores = []  
            # Loop over the range of k values
            for k in ks:
                # Fit the KMeans model for each k
                km = KMeans(n_clusters=k, random_state=42)
                # Fit the model to the standardized data
                km.fit(X_std)
                # Append the WCSS (inertia) to the list
                wcss.append(km.inertia_)  # inertia: sum of squared distances within clusterS
                # Get the cluster labels for the current k
                labels = km.labels_
                # Calculate the silhouette score for the current k
                silhouette_scores.append(silhouette_score(X_std, labels))
            # Create a title for the elbow plot
            st.write("##### Elbow Plot:")
            # Create a description of the elbow plot
            st.write("This plot shows the Within-Cluster Sum of Squares (WCSS) for different values of k. The 'elbow' point indicates the optimal number of clusters.")
            # Initiates the elbow plot
            elbow = plt.figure(figsize=(12, 5))
            # Initiates the elbow subplot
            plt.subplot(1, 2, 1)
            # Plot the WCSS result
            plt.plot(ks, wcss, marker='o')
            # Creates the x axis label for the elbow plot
            plt.xlabel('Number of clusters (k)')
            # Creates the y axis label for the elbow plot
            plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
            # Creates the title for the elbow plot
            plt.title('Elbow Method for Optimal k')
            # Creates the grid for the elbow plot
            plt.grid(True)
            # Displays the elbow plot in the app
            st.pyplot(elbow)

            # Create a title for the silhouette plot
            st.write("##### Silhouette Score:")
            # Create a description of the silhouette plot
            st.write("This plot shows the silhouette score for different values of k. A higher silhouette score indicates better-defined clusters.")
            # Initiates the silhouette plot
            silhouette = plt.figure(figsize=(12, 5))
            # Initiates the silhouette subplot
            plt.subplot(1, 2, 2)
            # Plot the silhouette score result
            plt.plot(ks, silhouette_scores, marker='o', color='green')
            # Creates the x axis label for the silhouette plot
            plt.xlabel('Number of clusters (k)')
            # Creates the y axis label for the silhouette plot
            plt.ylabel('Silhouette Score')
            # Creates the title for the silhouette plot
            plt.title('Silhouette Score for Optimal k')
            # Creates the grid for the silhouette plot
            plt.grid(True)
            # Puts the plots together in a single figure
            plt.tight_layout()
            # Displays the silhouette plot in the app
            st.pyplot(silhouette)


    # What the model does if the user wants to run a Hierarchical Clustering model
    elif modeltype == "Hierarchical Clustering":
        # Gives the user instructions to adjust the model settings
        st.write("### Step 3) Adjust Model Settings")
        # Prompts the user to select the target variable
        targets = st.selectbox("Select target variable:", df.columns)
        # Drops the target variable from the dataframe to be used in the model. Assigns the rest of the data to features_df
        features_df = df.drop(columns = targets)
        # Prompts the user to select the number of clusters
        k = st.slider("How many clusters should there be?", 2, 10, step = 1)
        # Imports the StandardScaler to be used on the data
        from sklearn.preprocessing import StandardScaler
        # Creates a title for the analysis section
        st.write("### Step 4) Analyze the Results")
        # Ensure all features are numeric by encoding categorical variables
        if not np.issubdtype(features_df.dtypes, np.number):
            features_df = pd.get_dummies(features_df, drop_first=True)
        # Standardize the numeric features (centering and scaling)
        scaler = StandardScaler()
        # Fits the scaler to the data and transforms it
        X_std = scaler.fit_transform(features_df)
        # Imports the AgglomerativeClustering model to be used on the data
        from sklearn.cluster import AgglomerativeClustering
        # Runs the AgglomerativeClustering model based on the number of clusters selected by the user
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        # Fits the model to the data
        df["Cluster"] = agg.fit_predict(X_std)
        # Assigns the cluster labels to the data
        cluster_labels = df["Cluster"].to_list()
        # Step 4: Visualize the Clustering Results Using PCA
        st.write("##### Scatterplot with PCA:")
        # Creates a description of the scatterplot
        st.write("This scatterplot shows the Agglomerative Clustering results projected onto the first two principal components.")
        # Imports the PCA to be used on the data
        from sklearn.decomposition import PCA
        # Reduce the dimensions for visualization (2D scatter plot)
        pca = PCA(n_components=2)
        # Fits the PCA to the data and transforms it
        X_pca = pca.fit_transform(X_std)
        # Initiates the scatterplot
        scatterplot = plt.figure(figsize=(10, 7))
        # For each cluster, it plots the data points in that cluster with a different color
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=60, edgecolor='k', alpha=0.7)
        # Creates the x axis label for the scatterplot
        plt.xlabel('Principal Component 1')
        # Creates the y axis label for the scatterplot
        plt.ylabel('Principal Component 2')
        # Creates the title for the scatterplot
        plt.title('Agglomerative Clustering on Data (via PCA)')
        # Creates the legend for the scatterplot
        plt.legend(*scatter.legend_elements(), title="Clusters")
        # Creates the grid for the scatterplot
        plt.grid(True)
        # Displays the scatterplot in the app
        st.pyplot(scatterplot)
        # Imports linkage and dendrogram to be used on the data
        from scipy.cluster.hierarchy import linkage, dendrogram
        # Creates a title for the dendrogram
        st.write("##### Dedrogram:")
        # Creates a description of the dendrogram
        st.write("This dendrogram shows the hierarchical clustering of the data. The y-axis represents the distance between clusters.")
        # Creates the dendrogram using the linkage matrix
        Z = linkage(X_std, method="ward")
        # Assigns the target variable to units to be used in the dendrogram plot
        units = df[targets].to_list()
        # Initiates the dendrogram plot
        fig, ax = plt.subplots(figsize=(20, 7))
        # Creates the dendrogram plot with the labels
        dendrogram(Z, labels= units)
        # Displays the dendrogram plot in the app
        st.pyplot(fig)
    
    # What the model does if the user wants to run a Principal Component Analysis model    
    elif modeltype == "Principal Component Analysis":
        # Gives the user instructions to adjust the model settings
        st.write("### Step 3) Adjust Model Settings")
        # Prompts the user to select the target variable
        targets = st.selectbox("Select target variable:", df.columns)
        # Prompts the user to select the feature variable
        features = st.multiselect("Select target variable:", df.columns)
        # Ensures that the user selects at least two feature variables so that an error does not occur
        if len(features) < 2:
            # If the user does not select at least two feature variables, a warning is displayed
            st.warning("Please select at least two feature variables.")
        # If the user selects at least two feature variables, the model runs
        else:
            # Creates a title for the analysis section
            st.write("### Step 4) Analyze the Results")
            # Assigns the feature variables to X to be used in the model
            X = df[features]
            # Assigns the target variable to y to be used in the model
            y = df[targets] 
            # Converts feature variables to dummy variables
            X = pd.get_dummies(X, drop_first=True)
            # Imports the StandardScaler to be used on the data
            from sklearn.preprocessing import StandardScaler
            # Standardizes the numeric features so that they have a mean of 0 and a standard deviation of 1
            scaler = StandardScaler()
            # Fits the scaler to the data and transforms it
            X_std = scaler.fit_transform(X)
            # Imports the PCA to be used on the data
            from sklearn.decomposition import PCA
            # We reduce the data to 2 components for visualization and further analysis.
            pca = PCA(n_components= 2)
            # Fits the PCA to the data and transforms it
            X_pca = pca.fit_transform(X_std)
            # Calculates the explained variance ratio for each principal component
            explained_variance = pca.explained_variance_ratio_
            # Display the explained variance ratio in the app
            np.cumsum(explained_variance)
            # Creates a color palette for the scatterplot
            colors = ['navy', 'darkorange']

            # Creates a title for the biplot
            st.write("##### Biplot:")
            # Creates a description of the biplot
            st.write("This biplot shows the PCA scores and loadings. The arrows represent the direction and magnitude of each feature's contribution to the principal components.")
            # Assigns the loadings to the PCA components
            loadings = pca.components_.T
            # Scales the loadings for better visualization
            scaling_factor = 50.0  # Increased scaling factor by 5 times
            # Initiates the biplot
            biplot = plt.figure(figsize=(8, 6))
            # For each color, it plots the data points in that color with a different color
            for color, i, target in zip(colors, [0, 1], targets):
                plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=0.7,
                    label=target, edgecolor='k', s=60)
            # Plot the loadings as arrows
            for i, feature in enumerate(features):
                plt.arrow(0, 0, scaling_factor * loadings[i, 0], scaling_factor * loadings[i, 1],
                color='r', width=0.02, head_width=0.1)
                plt.text(scaling_factor * loadings[i, 0] * 1.1, scaling_factor * loadings[i, 1] * 1.1,  # Adjusted text position
                feature, color='r', ha='center', va='center')
            # Creates the x axis label for the biplot
            plt.xlabel('Principal Component 1')
            # Creates the y axis label for the biplot
            plt.ylabel('Principal Component 2')
            # Creates the title for the biplot
            plt.title('Biplot: PCA Scores and Loadings')
            # Creates the legend for the biplot
            plt.legend(loc='best')
            # Creates the grid for the biplot
            plt.grid(True)
            # Displays the biplot in the app
            st.pyplot(biplot)

            # Creates a title for the Scree plot
            st.write("##### Scree Plot:")
            # Creates a description of the Scree plot
            st.write("This plot shows the eigenvalues of each principal component. It visually represents the variance explained by each component, helping to determine the optimal number of components to retain for further analysis.")
            # Runs PCA on the standardized data to get the explained variance ratio
            pca_full = PCA(n_components=min(X_std.shape)).fit(X_std)
            # Assigns the cumulative variance to the explained variance ratio
            # Calculates the cumulative variance explained by each principal component
            cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
            # Initiates the Scree plot
            scree = plt.figure(figsize=(8, 6))
            # Plots the cumulative variance explained by each principal component
            plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
            # Creates the x axis label for the Scree plot
            plt.xlabel('Number of Components')
            # Creates the y axis label for the Scree plot
            plt.ylabel('Cumulative Explained Variance')
            # Creates the title for the Scree plot
            plt.title('PCA Variance Explained')
            # Adds ticks to the x axis
            plt.xticks(range(1, len(cumulative_variance)+1))
            # Creates the grid for the Scree plot
            plt.grid(True)
            # Displays the Scree plot in the app
            st.pyplot(scree)

            # Creates a title for the Bar plot
            st.write("##### Bar Plot:")
            # Creates a description of the Bar plot
            st.write("This bar plot shows the variance explained by each principal component.")
            # Initiates the Bar plot
            bar = plt.figure(figsize=(8, 6))
            # Assigns the components to the range of the explained variance ratio
            components = range(1, len(pca_full.explained_variance_ratio_) + 1)
            # Initiates the bar plot
            plt.bar(components, pca_full.explained_variance_ratio_, alpha=0.7, color='teal')
            # Creates the x axis label for the Bar plot
            plt.xlabel('Principal Component')
            # Creates the y axis label for the Bar plot
            plt.ylabel('Variance Explained')
            # Creates the title for the Bar plot
            plt.title('Variance Explained by Each Principal Component')
            # Adds ticks to the x axis
            plt.xticks(components)
            # Creates the grid for the Bar plot
            plt.grid(True, axis='y')
            # Displays the Bar plot in the app
            st.pyplot(bar)




