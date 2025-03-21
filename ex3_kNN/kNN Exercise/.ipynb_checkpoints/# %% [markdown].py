# %% [markdown]
# 
# # Exercise: $k$-NN Classifier
# 
# In this exercise you will implement  the **$k$-Nearest Neighbor classifier ($k$-NN)**. You will also get familiar with
# other very important concepts related to machine learning in practice,
# including data preprocessing, distance metrics, visualization, and model evaluation.
# 
# We have provided general functionality and pointers for you here. Please complete the code with your own implementation below. Please also discuss and answer the follow-up questions.
# 
# ### 1. Dataset and problem description
# 
# The Healthy Body dataset contains body measurements acquired from **1250 people _from different ages, genders, and nationalities_** from different hospitals around the world. Health professionals have performed medical examinations and classified the individuals into three different body categories: **underweight, normal weight, and overweight.**
# 
# Our goal is to automate the role of the health professionals i.e, to predict the category of the new data . However, due to anonymity reasons, we have been provided access to limited information about the individuals: their measured _weights_ and _heights_, and their respective _body category_ only.
# 
# We will use these features to train a $k$-NN classifier for the task.
# 
# ---

# %%
# Enable interactive plots, so you can zoom/pan/resize plots
%matplotlib inline

# Libraries for numerical handling and visualization
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(0)

# %% [markdown]
# ## 2. Data loading and visualization
# 
# The goal of supervised classification algorithms such as $k$-NN is to use information from a set of labeled examples, i.e., examples for which we know their class assignments, to infer the classes for unlabeled examples, i.e., test examples for which we don't know their class.

# %%
# Weights, heights of individuals and their associated body category 
features_annotated_path = "./data/hbody_feats_annotated.npy"     
labels_annotated_path   = "./data/hbody_labels_annotated.npy"      

# Weights and heights of  individuals with unknown body category 
# Task: Figure out their body category label using k-NN classifier
features_unannotated_path = "./data/hbody_feats_unannotated.npy" 

# ground-truth body categories of  individuals with unknown body category  
# to evaluate the k-NN classifier
labels_unannotated_path = "./data/hbody_labels_unannotated_secret.npy"     

# Features organized in an NxD matrix: N examples and D features.
# Another way to look at it: each of the N examples is a D-dimensional feature vector.

data_train   = np.load(features_annotated_path)
data_test    = np.load(features_unannotated_path)
labels_train = np.load(labels_annotated_path)
labels_test  = np.load(labels_unannotated_path)

class_names = ('Underweight', 'Normal weight', 'Overweight')


# %% [markdown]
# **Q1. What are our labels? What are the features that we use to predict these labels?**
# 
# **Answer:** our labels are 'Underweight', 'Normal weight' and 'Overweight', and our features that we use to predict these labels are weights and heights of the individuals.
# 
# 

# %% [markdown]
# ### 2.1 Data Summary

# %%
print("Number of examples in the training set  : {}".format(data_train.shape[0]))
print("Number of examples in the test set      : {}".format(data_test.shape[0]))

plt.bar(class_names, np.bincount(labels_train))
plt.ylabel('Count')
plt.title('Histogram plot of each body category in the training set');

# %% [markdown]
# ### 2.2 Data visualization

# %%
colors = np.array([[1.0, 0.0, 0.0], 
                   [0.0, 1.0, 0.0], 
                   [0.0, 0.0, 1.0]])

plt.figure(figsize=(16,6))

# visualize the training set
plt.subplot(1,2,1)
plt.title(f"Training set ({len(labels_train)} examples)")
for i, class_name in enumerate(class_names):
    plt.scatter(*data_train[labels_train==i].T,
                c=colors[i, None], alpha=1.0, 
                s=20, lw=0, label=class_name)
plt.xlabel("Weight (kg)")
plt.ylabel("Height (cm)")
plt.legend();

# visualize the test set
plt.subplot(1,2,2)
plt.title(f"Test set ({len(labels_test)} examples)")
for i, class_name in enumerate(class_names):
    plt.scatter(*data_test[labels_test==i].T,
                c=colors[i, None], alpha=1.0, 
                s=20, lw=0, label=class_name)
    
plt.xlabel("Weight (kg)")
plt.ylabel("Height (cm)")
plt.legend();
plt.show()

# %% [markdown]
# **Q2. Do you think this is an easy or a difficult classification problem? Why?**  
# 
# **Answer:** I think the classification problem is easy due to the relativly distinct scattering between data from different classes. Although it's worth noticing that there are some outliers (for example: underweight classified data points scattered among normal weight data points.
# 
# **Q3. What should the test set share in common with the training set?**  
# 
# **Answer:** The test set should share the same distribution as the training data. 
# 
# 

# %% [markdown]
# ## 3. Normalizing the data
# 
# k-NN determines neighbors by computing the "distance" between two examples. For this process to work, we are required to normalize the features. This is true for many other machine learning algorithms as well!
# 
# **Q. What would happen if we don't do this?**  
# **A.**  If we won't normalize the features, some features will have a greater impact on the distance calculated, therefore will unintentionally impacting more on the end prediction.
# 
# A common way to normalize the data is by the so-called z-score standardization. It transforms values from an arbitrary range such that the results have mean $0$ and standard deviation $1$. 
# 
# The operation is defined as follows:
# 
# $$
# x_{norm} = \frac {x - \mu_x} {\sigma_x},
# $$
# 
# and is computed _independently for each feature_ $x$ using its mean $\mu_x$ and standard deviation $\sigma_x$.
# 
# Thanks to NumPy however, we can parallelize this by operating on arrays. Pay attention to the dimensions below, taking care that the full data matrix $X$ is $N\times D$ ($N$ rows of $D$-dimensional vectors).

# %%
def normalize(data, means, stds):
    """This function takes the data, the means,
    and the standard deviatons (precomputed), and 
    returns the normalized data.
    
    Inputs:
        data : shape (NxD)
        means: shape (1XD)
        stds : shape (1xD)
        
    Outputs:
        normalized data: shape (NxD)
    """
    # return the normalized features
    # WRITE YOUR CODE HERE
    normalized_data = np.zeros(data.shape)
    for row, feature in enumerate(data):
        normalized_data[row] = ((feature - means) / stds)
    return normalized_data

# test the  normalize function 
dummy_features = np.random.randint(100,size=(10,3))
means = dummy_features.mean(0,keepdims=True)
stds  = dummy_features.std(0,keepdims=True)
dummy_features_normed = normalize(dummy_features, means, stds)

if (np.allclose(dummy_features_normed.mean(axis=0), 0) and 
    np.allclose(dummy_features_normed.std(axis=0), 1)):
    print("Everything alright here!!")
else:
    print("Try again.")

# %% [markdown]
# ## 4. The $k$-Nearest Neighbors Classifier
# 
# The k-NN classifier assigns the most common label among its k-nearest neighbors for a given example. The method is very intuitive and can be summarized as:
# - Compute the distance between the example to classify and all the training examples.
# - Select the closest $k$ training examples.
# - Assign to the example the most common label among those neighbors.
# 
# ### 4.1 Distance metrics
# 
# There are many ways to define a distance between two examples. You are probably used to a very common distance metric, the Euclidean distance. 
# 
# #### Euclidean distance:
# 
# $$
# D(\mathbf{v}, \mathbf{w}) = \sqrt{ \sum_{i=1}^D \left(\mathbf{v}^{(i)} - \mathbf{w}^{(i)}\right)^2 }
# $$
# 
# This distance metric corresponds to our intuitive interpretation of the straight-line distance between two vectors $\mathbf{v}\in\mathbf{R}^D$ and $\mathbf{w}\in\mathbf{R}^D$. Note that $\mathbf{v}^{(i)}$ denotes the value in the dimension $i$ of $\mathbf{v}$.
# 
# 

# %%
def euclidean_dist(example, training_examples):
    """Compute the Euclidean distance between a single example
    vector and all training_examples.

    Inputs:
        example: shape (D,)
        training_examples: shape (NxD) 
    Outputs:
        euclidean distances: shape (N,)
    """
    # WRITE YOUR CODE HERE
    sum_of_distances = 0
    for row in training_examples:
        sum_of_distances += np.power(row - example,2)
    
    return np.sqrt(sum_of_distances)

# %%
def find_k_nearest_neighbors(k, distances):
    """ Find the indices of the k smallest distances from a list of distances.
        Tip: use np.argsort()

    Inputs:
        k: integer
        distances: shape (N,) 
    Outputs:
        indices of the k nearest neighbors: shape (k,)
    """
    # WRITE YOUR CODE HERE
    indices = np.argsort(distances)[0:k]
    return indices

# test the find_k_nearest_neighbors function
dummy_distances = [10., 0.5, 200, 0.006, 43, 4.5, 11., 50]
top_k = 5
top_k_indices = find_k_nearest_neighbors(top_k,dummy_distances)

if np.allclose(top_k_indices,[3,1,5,0,6]):
    print('Implementation is correct')
else:
    print('Oops!! Something is wrong')
        

# %%
# Given a list of neighbor labels, choose the most frequent one.
# Tip: np.bincount and np.argmax are your friend.

def predict_label(neighbor_labels):
    """Return the most frequent label in the neighbors'.

    Inputs:
        neighbor_labels: shape (N,) 
    Outputs:
        most frequent label
    """
    # WRITE YOUR CODE HERE
    return np.argmax(np.bincount(neighbor_labels))

# test label prediction
dummy_labels = [10, 3, 2, 10, 2, 2, 2, 10, 10, 11, 1, 2]

if predict_label(dummy_labels) == 2:
    print('Implementation is correct')
else:
    print('Oops!! Something is wrong')



# %% [markdown]
# ### 4.2 $k$-NN classifier, step by step for a single test example
# 
# Let's implement the algorithm for **one data sample**, i.e., to classify a single new point.
# 
# **Q4. You are asked to use the mean and std of the training set to normalize the test data. Can you explain the reasoning behind this?** 
# 
# **Answer:** As I've explained previously, the reasoning behind normalization of the test data using the mean and std of the training set is that they both have the same distribution, therefore by normalizing using the mean and std of the training set we maintain the consistency between the two.
# 
# 
# 

# %%
def choose_random_sample(data):
    """Randomly chose a single datapoint from the given data matrix.
    
    Input:
        data: shape (NxD)
    Output:
        randomly chosen sample: shape (1xD)
    """
    total_samples = data.shape[0]
    rand_idx = np.random.choice(total_samples)
    return data[rand_idx : rand_idx + 1]

# %%
# Use the functions you defined above to predict the label of a single example-

# First, we choose a random test example 
single_test_data =  choose_random_sample(data_test)

# IMPORTANT: Normalize the data, what should the mean and std be?
# Hint: mean_val and std_val dimension should be (1x2). Make use of reshape or the argument `keepdims`.

# WRITE YOUR CODE HERE
mean_val = np.mean(data_train, axis=0, keepdims=True)
std_val = np.std(data_train, axis=0, keepdims=True)

# norm_train is the normalized data_train
norm_train = normalize(data_train, mean_val, std_val)

# norm_test_single is the normalized single_test_data
norm_test_single = normalize(single_test_data, mean_val, std_val)

# set the number of neighbors in the kNN classifier (hyperparameter)
# (you can play with that)
k = 2

# reshape the single example from (1xD) to (D,) for the next function
norm_test_single = norm_test_single.reshape(-1)
# find distance of the single test example w.r.t. all training examples
distances = euclidean_dist(norm_test_single, norm_train)

# find the nearest neighbors
nn_indices = find_k_nearest_neighbors(k, distances)

# find the labels of the nearest neighbors
neighbor_labels = [labels_train[i] for i in nn_indices]

print("Nearest {} neighbors' labels: {}\n".format(k, ", ".join([class_names[x] for x in neighbor_labels])))

# find the best label
best_label = predict_label(neighbor_labels) 

print(f'Predicted label: {class_names[best_label]}')

# %% [markdown]
# ##### The cell below is for visualization.

# %%
# Visualize the unknown point and its neighbors.
plt.figure(figsize=(8, 6))

plt.title(f"A randomly chosen unlabeled example from the "
          "test set\nand its k-nearest neighbors")

for i, class_name in enumerate(class_names):
    plt.scatter(*norm_train[labels_train==i].T,
                c=colors[i, None], alpha=0.25, 
                s=20, lw=0, label=class_name)
    
for i, class_name in enumerate(class_names):
    class_indices = nn_indices[labels_train[nn_indices] == i]
    if len(class_indices) > 0:
        plt.scatter(*norm_train[class_indices].T,
                    c=colors[i, None], alpha=1, 
                    s=25, lw=0, label='Neighbor')

# Make sure the norm_test is 1D vector.
plt.scatter(*norm_test_single.reshape((-1)), marker='*', c='darkorange', 
                 alpha=.9, s=75, label='unlabeled example')

plt.xlabel("Weight (normalized)")
plt.ylabel("Height (normalized)")
plt.legend();

# %% [markdown]
# ### 4.3 Performance metrics
# 
# To quantify the performance of our model, we want to obtain a score that tells us how close the predictions were to the expected classification.
# 
# The simplest way to do this is to compute the ratio of correctly predicted examples, also known as the accuracy:
# 
# $$
# \frac 1 N \sum_{n=1}^N \mathbf{1}[\hat{y} = y]
# $$
# 
# where the indicator function $\mathbf{1}[\hat{y} = y]$ returns 1 if the predicted $\hat{y}$ is equal to the ground-truth $y$ and 0 otherwise.
# 
# **Q5. Do you see any limitation to using accuracy to evaluate your model?**
# 
# **Answer.** High accuracy might not indicate that our model works for every data set. To add to it, accuracy assumes all errors are equally costly, which may not hold in real-world scenarios (for example: misclassifying a cancer-positive patient as negative is more serious than the reverse).  
# 
# **Q6. Can you think of other ways to evaluate your model?**
# 
# **Answer.** We could evaluate the error rate of the model.
# 
# **Q7. What other criteria, aside from accuracy, should one consider when choosing hyperparameters?**  
# 
# **Answer.** Error rate.
# 
# 
# 

# %%
# Write a function that computes the accuracy between a predicted and the expected labels.
def compute_accuracy(predicted, target):
    """Returns the accuracy score.

    Inputs:
        predicted: shape (N,) 
        target: shape (N,) 
    Outputs:
        accuracy
    """
    # WRITE YOUR CODE HERE
    N = target.shape[0]

    correct_predictions = np.sum(predicted == target)
    return correct_predictions / N

# %% [markdown]
# ### 4.4 Putting things together to run the k-NN classifier for all examples
# 
# Let's implement the algorithm for **all data samples**.
# 
# 

# %%
# Write a function kNN_one_example that applies all the previous steps
# to predict the label of 'one' example.

def kNN_one_example(unlabeled_example, training_features, training_labels, k):
    """Returns the label of a single unlabelled example.

    Inputs:
        unlabeled_example: shape (D,) 
        training_features: shape (NxD)
        training_labels: shape (N,) 
        k: integer
    Outputs:
        predicted label
    """
    # WRITE YOUR CODE HERE
    mean_value = np.mean(training_features)
    std_value = np.std(training_features)

    norm_train_data = normalize(training_features, mean_value, std_value)
    norm_example_data = normalize(unlabeled_example, mean_value, std_value)

    # Compute distances
    distances = euclidean_dist(unlabeled_example, training_features)
    
    # Find neighbors
    nn_indices = find_k_nearest_neighbors(k, distances)
    
    # Get neighbors' labels
    neighbor_labels = [training_labels[i] for i in nn_indices]
    
    # Pick the most common
    best_label = predict_label(neighbor_labels)
    
    return best_label

# %%
# Write a function kNN that applies 'kNN_one_example' function to an arbitrary number of examples.
# Tip: NumPy's apply_along_axis does most of the work for you. 
# It's a one-liner, but you might need to read its documentation!

def kNN(unlabeled, training_features, training_labels, k):
    """Return the labels vector for all unlabeled datapoints.

    Inputs:
        unlabeled: shape (MxD) 
        training_features: shape (NxD)
        training_labels: shape (N,) 
        k: integer
    Outputs:
        predicted labels: shape (M,)
    """
    # WRITE YOUR CODE HERE
    return np.apply_along_axis(kNN_one_example, 1, unlabeled, training_features, training_labels, k)

# %%
# Normalize your training data and test data
# (Don't forget to normalize according to the mean and std of the training set!)

mean_val = np.mean(data_train, axis=0)
std_val  = np.mean(data_train, axis=0)
norm_train_data = normalize(data_train, mean_val, std_val)
norm_test_data  = normalize(data_test, mean_val, std_val)

# choose a k value (you can play with that)
k = 6

# run k-NN classifier on complete test data
predicted_labels_test = kNN(data_test, data_train, labels_train, k)

accuracy = compute_accuracy(predicted_labels_test, labels_test)
print("Test Accuracy is {:.1f}%".format(100*accuracy))

# Visualize the predictions on the unannotated test set
plt.figure(figsize=(8,6))
plt.title("Predicted classes for test data")

for i, class_name in enumerate(class_names):
    plt.scatter(*norm_train_data[labels_train==i].T,
                c=colors[i, None], alpha=0.1, s=15, lw=0)
    
# represent test set by '*' marker
for i, class_name in enumerate(class_names):    
    plt.scatter(*norm_test_data[predicted_labels_test==i].T,
                c=colors[i, None], marker='*', alpha=0.7, 
                s=50, lw=0, label=class_name)
    
plt.xlabel("Weight (normalized)")
plt.ylabel("Height (normalized)")
plt.legend();

# %% [markdown]
# **Q8. How do you expect the model to perform with large k values equal to the number of training example ?**
# 
# **Answer.** 
# 
# **Q9. While the above implementation works, it has some drawbacks. Can you identify them?**  
# 
# **Answer.** 
# 
# **Q10. What should one take into account when feeding new data to a machine learning model?**
# 
# **Answer.** 

# %% [markdown]
# ## 4.5 Written questions

# %% [markdown]
# **Q1.** You use $k$-NN on a dataset with $k=21$. You find out that it overfits as performance on the training set is a lot better than on the validation set. What do you expect might help to reduce overfitting: decreasing or increasing the number of neighbors $k$?
# 
# **A1.** 

# %% [markdown]
# **Q2.** Class imbalance in the training set can be problematic for $k$-NN. Why? What potential solutions could help?
# 
# **A2.** 

# %% [markdown]
# **Q3.** (MCQ) Which of the following statements are true for $k$-NN?
# 1. For $k=100$, the time to train $k$-NN is longer than for $k=10$.
# 2. $k$ should be set to the number of classes.
# 3. $k$-NN has _no_ trainable parameters.
# 4. The prediction of the algorithm might change with the distance metric.
# 
# **A3.** 

# %% [markdown]
# **Q4.** (SCQ) Let us consider the 2-dimensional training data in the figure below, which contains two classes: "Positive" and "Negative", denoted by the symbols $+$ and $-$ respectively. Furthermore, let us consider the cosine distance between two points $\mathrm{u}$ and $\mathrm{v}$ given by
# 
# $$ d_{cosine}(\mathrm{u}, \mathrm{v}) = 1 - \frac{\mathrm{u}^\top\mathrm{v}}{\lVert\mathrm{u}\rVert_2 \cdot \lVert\mathrm{v}\rVert_2} = 1 - \cos\theta, $$
# 
# where $\theta$ is the angle between the vector $\mathrm{u}$ and $\mathrm{v}$.
# 
# ![](img/knn_question.png)
# 
# Which of the following statement is the correct one if we predict the label of the test point with a $1$-NN?
# 1. Using both cosine and euclidean distances, it is classified as "Positive".
# 2. It is classified as "Positive" with the cosine distance and "Negative" with euclidean distance.
# 3. Using both cosine and euclidean distances, it is classified as "Negative".
# 4. It is classified as "Negative" with the cosine distance and "Positive" with euclidean distance.
# 
# **A4.** 

# %% [markdown]
# ## 5 Cross Validation (Optional)
# 
# How should we be choosing the value of k? If we choose the k that gives us the highest **test** accuracy, we would be **cheating**, because we would be tuning our model to its test performance. 
# 
# In practice, we choose the k that gives us the highest **validation** accuracy, via the cross-validation method. By doing so, we ensure that we select a method which generalizes well to unseen test data.

# %% [markdown]
# ### 5.1.  K-Fold Cross Validation 
# In the course, you saw that we generally reserve a portion of the training data for validation, i.e., testing the performance of our model to choose the hyper-parameter. Below, we introduce "K-Fold Cross-Validation".
# 
# K-fold is a type of cross validation technique that works in the following way:
# 
# 1 - Select a $k$ (do not confuse this with the K-Fold of the cross validation, this is the $k$ for the kNN method!)
# 
# 2 - Split the training data in K equal and disjoint parts
# 
# 3 - Select 1 part as our validation set and the rest as our training set.
# 
# 4 - Train our model on our training set and find the accuracy of the validation set. 
# 
# 5 - Repeat steps 3 and 4 K times, each time selecting a different part of the data for the validation set. In the end we will find K different validation accuracies. We will average the validation accuracies and find the validation accuracy that corresponds to the k we chose for kNN. (See the image below).
# 
# 6 - Repeat steps 1-5 (the whole process) for different $k$ values (hyperparameter for kNN). 
# 
# 7 - Find the $k$ value that gave the highest validation accuracy. Train your model on the whole training set using this $k$. Test on the test set and report the test accuracy.
# 

# %% [markdown]
# **Q. What is the difference between validation set and test set?**  
# **A.** 
# 
# 

# %% [markdown]
# ![](img/cross_validation.png)
# 
# 

# %% [markdown]
# ### 5.2. Implementation
# 
# Now let's begin! We will be doing steps 1-5 of our algorithm above. We will do a K-Fold cross validation to find the validation accuracy for a selected $k$.

# %%
def KFold_cross_validation_KNN(X, Y, K, k):
    '''
    K-Fold Cross validation function for K-NN

    Inputs:
        X : training data, shape (NxD)
        Y: training labels, shape (N,)
        K: number of folds (K in K-fold)
        k: number of neighbors for kNN algorithm (the hyperparameter)
    Returns:
        Average validation accuracy for the selected k.
    '''
    N = X.shape[0]
    
    accuracies = []  # list of accuracies
    for fold_ind in range(K):
        #Split the data into training and validation folds:
        
        #all the indices of the training dataset
        all_ind = np.arange(N)
        split_size = N // K
        
        # Indices of the validation and training examples
        val_ind = all_ind[fold_ind * split_size : (fold_ind + 1) * split_size]
        ## YOUR CODE HERE (hint: np.setdiff1d is your friend)
        train_ind = ...
        
        X_train_fold = X[train_ind, :]
        Y_train_fold = Y[train_ind]
        X_val_fold = X[val_ind, :]
        Y_val_fold = Y[val_ind]

        # Run KNN using the data folds you found above.
        # YOUR CODE HERE
        Y_val_fold_pred = ...
        acc = ...
        accuracies.append(acc)
    
    #Find the average validation accuracy over K:
    ave_acc = ...
    return ave_acc

# %%
# Perform a 4-fold cross validation using k-NN, with k=5.

K = 4 # 4 fold cross validation
k = 5 # number of nearest neighbours for k-NN
acc = KFold_cross_validation_KNN(norm_train_data, labels_train, K, k)
print(f"{k}-NN Classifier predicted with average validation accuracy of {acc:.2%}.")

# If your accuracy is not above 80%, you may have made a mistake.

# %% [markdown]
# ### 5.3 Hyperparameter optimization
# 
# Now let's find the best $k$! Run the function you wrote above for different values of $k$, and let's see which one gives the best validation accuracy.
# 

# %%
# Call the cross validation method with different k values

def run_cv_for_hyperparam(X, Y, K, k_list):
    '''
    K-Fold Cross validation function for K-NN

    Inputs:
        X : training data, shape (NxD)
        Y: training labels, shape (N,)
        K: number of folds (K in K-fold)
        k: a list of k values for kNN 
    Returns:
        model_performance: a list of validation accuracies corresponding to the k-values     
    '''
    model_performance = [] 
    for k in k_list:
        #YOUR CODE HERE
        ...
    return model_performance


# %%
# Visualize the performances for different values of k

# Try these values for hyperparameter k
k_list = range(1, 100, 4)          
K = 4 # number of fold for K-Fold

## YOUR CODE HERE
model_performance= run_cv_for_hyperparam(norm_train_data,labels_train, K, k_list)

plt.figure(figsize=(9,4))
plt.title("Performance on the validation set for different values of $k$")
plt.plot(k_list, model_performance)
plt.xlabel("Number of nearest neighbors $k$")
plt.xticks(k_list)
plt.ylabel("Performance (accuracy)");

# %%
# Pick hyperparameter value that yields the best performance
# WRITE YOUR CODE HERE
best_k = ...

print(f"Best number of nearest neighbors on validation set is k={best_k}")

# %% [markdown]
# ### 5.4. Test accuracies for best model.
# 
# Now that we have tuned our model, we can apply it for prediction on the test set using the optimal $k$ found on cross-validations set.
# 
# **Q. How do you expect the model to perform, compared with the cross-validation set performance?**
# 
# **A.** 

# %%
# Get predicted labeles for unannotated data.
# WRITE YOUR CODE HERE
predicted_labels_test = ...
accuracy = compute_accuracy(predicted_labels_test, labels_test)
print("Test Accuracy is {:.1f}%".format(100*accuracy))

# %%

# Visualize the predictions on the unannotated test set
plt.figure(figsize=(8,6))
plt.title("Predicted classes for test data")

for i, class_name in enumerate(class_names):
    plt.scatter(*norm_train_data[labels_train==i].T,
                c=colors[i, None], alpha=0.1, s=15, lw=0)
    
# represent test set by '*' marker
for i, class_name in enumerate(class_names):    
    plt.scatter(*norm_test_data[predicted_labels_test==i].T,
                c=colors[i, None], marker='*', alpha=0.7, 
                s=50, lw=0, label=class_name)
    
plt.xlabel("Weight (normalized)")
plt.ylabel("Height (normalized)")
plt.legend();


