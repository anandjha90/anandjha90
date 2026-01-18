## AI interview questions   
  
**Question1:**  What is the relation / difference between AI, ML, DL, NLP  
AI (Artificial Intelligence) : Its an application, it performs its own tasks without any human intervention.   
Ex: Robots, self driving cars, Recommendation Systems etc  
If we consider entire world as AI,   
Machine Learning is a subset of AI.  
Deep Learning is a subset of Machine Learning.  
NLP is the part of Machine Learning and Deep Learning.  
Now, let us try to understand, where Data Science comes into the picture.  
Data Science people comes into the all vertices.  
Example:  
If the person is working on Machine Learning, we can call him / her as Machine Learning Engineer which is part of Data Science.  
Similarly, if the person is working on Deep Learning / NLP, we can call him / her as Deep Learning / NLP Engineer.  
Hope you got some idea.  

**Question 2: What is Supervised Machine Learning?**  
**Ans: **Supervised Machine Learning algorithms are trained on labelled data (which consists of rows and columns in simple words). In this we have dependent (Input features) and independent features (Output feature / Target feature)  
**Examples:**  
Spam Filtering, Loan prediction, Customer Churn Prediction etc.  

**Question 3: What is Unsupervised Machin Learning Algorithm?**  
Ans: In unsupervised Machine Learning, we make / form a clusters / groups based on similar characteristics.  
These algorithms will find the hidden patterns of the given dataset and forms a cluster.  
It trained with unlabelled data but here we can’t find dependent and independent features.  
**Example:** Customer segmentation, Fraud detection etc  

**Question 4: What is Semi-supervised Machine Learning?**  
**Ans: **This approach combines both of Labeled and unlabeleddata to train the model.  
This combination can lead to models that generalize better and achieve higher accuracy compared to using only labeleddata.   
Example: Text classification, Weather forecasting, Image classification and many more.  

**Question 5: What is Reinforcement learning (RL)**  
Ans: It’s a paradigm where an agent learns to make decisions in an environment to maximize a reward signal. It's a trial-and-error process where the agent interacts with its environment, takes actions, and receives feedback in the form of rewards or penalties.   
The goal is for the agent to learn an optimal policy, which is a mapping of states to actions, that maximizes the cumulative reward over time.   
**Key Concepts:**  
• **Agent:** The entity that interacts with the environment and makes decisions.   
• **Environment:** The context in which the agent operates, including its current state and the consequences of the agent's actions.   
• **State:** The current situation of the environment, which influences the agent's decision-making.   
• **Action:** The choices the agent can make in response to the current state.   
• **Reward:** Feedback received by the agent after taking an action, indicating whether the action was beneficial or not.   
• **Policy:** The strategy the agent uses to choose actions based on the current state.   
![The general framework of](Attachments/9409CA57-00F1-4366-94C1-2D7B91ED015C.png)  
**Applications:**  
Robotics, Game Playing, Resource Management and more.  

**Question 6: What is Gradient descent**  
Ans: Gradient descent in machine learning is simply used to find the values of a function's parameters (coefficients) that minimize a cost function as far as possible.  
![Gradient](Attachments/94A797ED-B0D3-418D-886C-1A9FD2866237.png)  

**Question 7: What is Classification problem statement?**  
**Ans: **This is one of the supervised Machine Learning Algorithm’s**. **Here we will be having independent (Input) and dependent (Output) features.  
In the classification problem we will be having limited number of classes / outcomes in the dependent (output) feature.  
**Case 1:** If we are having two classes / outcomes in the output, we can say that as Binary Classification problem statement.Ex: "yes/no," "true/false," or "positive/negative"  
**Case 2:** If we have more than 2 classes / outcomes, we can say that as Multiclass Classification problem statement. Ex: Good/  
**Example:**  
• Classifying handwritten digits from 0 to 9,   
• Classifying different types of fruits (like apple, banana, orange, etc.) from images  
• Sentiment analysis as Good, Bad, Average   

**Question 8: What is regression problem statement?**  
Ans: This is one of the supervised Machine Learning Algorithm’s**. **Here we will be having independent (Input) and dependent (Output) features.  
In the output variable we will be having continuous values.  
**Ex:** House price prediction, Predicting the sales revenue, Car Price prediction.  
   
**Question 9: Is there any difference between cost and loss functions?**  
Ans: The key difference between a cost function and a loss function lies in their scope:   
a loss function quantifies the error for a single data point, while a cost function calculates the average or total loss across an entire dataset.  
   
**Question 10: What are the cost functions of linear regression?**  
Ans:   
• Mean Squared Error (MSE),   
•   
![MSE = 1=0](Attachments/498EAE80-BEBD-4939-A30F-2B803F1ED342.png)  
• Root Mean Squared Error (RMSE), and  
• Mean Absolute Error (MAE).   
•   
![MAE = 1=0](Attachments/918643FD-0403-40CB-8EA4-8F00D12A869A.png)  
The goal is to minimize the cost function to improve the model's accuracy and make better predictions.   

**Question 11: What are the cost functions in Classification problems?**  
**Ans: Common cost function includes **cross-entropy (also known as log loss)  
• Binary Cross Entropy                - Binary classification   
• Categorical Cross Entropy and – Multiclass classification  
• Sparse Categorical Cross Entropy – Multiclass classification  

**Question 12: What are the performance metrics of Classification?**  
Ans:   
• Accuracy -   (TP + TN) / (TP + TN + FP + FN)  
• Precision - TP / (TP + FP)  
• Recall - TP / (TP + FN)  
• F1-score - 2 * (Precision * Recall) / (Precision + Recall) and  
• AUC-ROC (Area Under the Curve – Receiver Operating Characteristic curve)  

**Question 13: What are the performance metrics for regression**  
• **R-squared and **  
• **Adjusted R-Squared.**  

**Question 14: What is overfitting how to avoid that?**  
**Ans:**  
**Overfitting occurs when a Machine Learning model performs well on training data and less well on testing data.**  
**In Otherwards: The scenario where we can reach, low bias and high variance.**  
**How to avoid:**  
**Using cross-validation, regularization, data augmentation, early stopping, and simplifying the model can be employed techniques, we can avoid overfitting condition**  

**Question 15: What is underfitting?**  
Ans: It is opposite to the Overfitting condition where the Machine Learning model performs more well on testing data, less well on training data.  
Underfitting is characterized by high bias and low variance.   
**How to avoid:**  
• Increase Model Complexity  
• Adjust Training Parameters  
• Enhance Feature Engineering  
• Reduce regularization  
   
**Question 16: What is multicollinearity?**  
Ans. Multicollinearity refers to a situation where two or more independent variables in a model are highly correlated, making it difficult to determine the individual effect of each variable on the dependent variable  

**Question 17: What is homoscedasticity?**  
Ans: In Machine Learning, especially in linear regression models, homoscedasticity refers to the assumption that the variance of the error terms (residuals) is constant across all levels of the independent variables. 

**Question 18: What is mean by residual?**  
Ans. In Machine Learning, a residual refers to the difference between the actual (observed) value and the predicted value from a model  

**Question 19: What are the assumptions in simple linear regression?**  
**Ans:**  
• Linearity: he relationship between the independent and dependent variables must be linear.  
• Independence of Errors: The errors (residuals) for each observation must be independent of each other.  
• Homoscedasticity: The variance of the errors should be constant across all levels of the independent variable.  
• Normality of Errors: The errors (residuals) should be normally distributed  

**Question 20: Can you explain Data Science Life Cycle?**  
**Ans: **  
1. Business Understanding: The complete cycle revolves around the enterprise goal.  
2. Data Understanding: Here you need to intently work with the commercial enterprise group as they are certainly conscious of what information is present, what facts should be used for this commercial enterprise problem, and different information.  
3. Preparation of Data: Format the data into the preferred structure, eliminate undesirable columns and features. Data preparation is the most time-consuming but arguably the most essential step in the complete existence cycle.  
4. Exploratory Data Analysis: Profiling of the data, Statistical Analysis, Graph based analysis.  
5. Data Modeling (Model Building/ Model Training): This is the coronary heart of data analysis. A model takes the organized data as input and gives the preferred output  
6. Model Evaluation: Evaluate the model using performance metrics.   
7. Model Deployment:** **The model after a rigorous assessment is at the end deployed in the preferred structure and channel.  
**++[https://media.geeksforgeeks.org/wp-content/uploads/20200805200955/datasc-660x434.png](https://media.geeksforgeeks.org/wp-content/uploads/20200805200955/datasc-660x434.png)++**
   
**Question 21. How do we impute the null / missing values?**  
**Ans: **  
• **Randomly filling**  
• **Forward filling**  
• **Backward filling**  
• **Using some statistical tools like mean, median and mode**  
• **KNN Imputer**  
• **End of the distribution**  
• **Remove the null values**  
• **Builds a own ML algorithms to impute the null values**  

**Question 22 : How do we find the Outliers and how can we handle them?**  
Ans:  
Ways to find the outliers:  
• sorting the data,   
• visualizing it with graphs (box plots, scatter plots, histograms)  
• calculating z-scores  
• using the Interquartile Range (IQR)  
Ways to Handle the outliers:  
• Removal  
• Replace with another other values   
• Transformation (Scaling (Standardization, normalization etc), Log Transformation, Winsorization)  

**Question 23: What is encoding? How many types of encodings we have?**  
**Ans: The process of converting the categorical data in to numerical form is called Encoding.**  
**Types:**  
• One Hot Encoding  
• Label Encoding  
• Hash Encoding  
• Binary Encoding  
• Target Encoding  

**Question 24: How do we perform Transformation?**  
Ans:   
• Log transformation  
• Square transformation  
• Cube transformation  
• Power transformation  

**Question 25: How do we perform Scaling?**  
**Ans: **  
• Standardization  
• Normalization  
• Unit Scaling  
**Question 26: What is Hyperparamer Tunning? **  
Ans : Hyperparameter tuning is the process of finding the optimal configuration of parameters for a machine learning model, which are called hyperparameters. These hyperparameters are set before the model training process begins, and they influence the model's learning process. By tuning these hyperparameters, you can optimize the model's performance and make it more accurate.   
Why do we perform it?  
• Improves Model Performance  
• Reduces Overfitting and Underfitting  
• Optimizes Training Process  
• Generalization to Unseen Data  

**Question 27: Types of Hyperparameter tuning?**  
• Manual Tuning:  
This involves manually trying different hyperparameter combinations and observing their impact on model performance.   
• Grid Search:  
This method systematically explores all combinations of hyperparameter values within a predefined range.   
• Random Search:  
Instead of exploring all combinations, random search randomly samples hyperparameter values from a specified distribution.   
• Bayesian Optimization:  
This method uses a probabilistic model to intelligently select the next set of hyperparameters to try, based on past evaluations.   

**Question 28: What is Decision Tree stump?**  
Ans: Decision Tree with only one depth is called stump. Or Decision Tree with One level  

**Question 29: What is leaf node in Decision Tree?**  
Ans: In a decision tree, from whichever node if further split is not possible, we can say that as lead node.   
In simple words, a node which has no child nodes is called lead node.  

**Question 30: What is Pure Split?**  
Ans. In decision tree algorithms, a "pure split" refers to a split (or division) of data that results in each resulting subset containing only instances of a single class.  
We can measure the purity of the split based on Entropy or Gini Indexing.  
• Entropy: *H*(*s*)=−*P*(+) log2 *P*(+) −*P*(−) log2 *P*(−)  
• Gini indexing: *GI*=1−∑*i*=1*n* (*pi* )2  
![HCS), GIS](Attachments/49025CD3-734B-4A53-9112-261FFFFE099C.png)  

**Question 31: What is impure split?**  
**Ans. **This is the opposite of a pure split, in which data is distributed among the different classes.  
Here we can measure the impurity values using Entropy or Gini Indexing.  

**Question 32: How do we find out whether that node / feature is root node or not?**  
**Ans: **In a decision tree, the root node is the topmost node and represents the entire dataset at the beginning of the tree.  
It's the starting point where the algorithm begins to make decisions based on the features or attributes of the data.  
How do we say that is root node:   
Whichever node is having the high information gain mathematically, we can consider that node as root node.  

**Question 33: What is Entropy and Gini impurity**  
**Ans: **In decision tree algorithms, both entropy and the Gini index are used to measure the impurity of a node.  
**A lower value indicates a more pure node, meaning the data within that node is more homogeneous.**  

**Question 34: What is the use of Information Gain?**  
Ans: Information gain in decision trees measures how much a particular feature reduces uncertainty about the target variable. It's a key metric used to determine which features are most useful for splitting data and building a decision tree.  
In simple words, based on this information Gain, we can be able to find the root node.  

**Question 35: What is CART Approach?**  
**Ans: **CART, short for Classification and Regression Trees, is a decision tree algorithm used in machine learning for both classification and regression tasks.  
Splitting Criteria:  
• Classification: Uses Gini impurity or information gain to determine the best split.   
• Regression: Uses mean squared error (MSE) or variance to determine the best split.   

**Question 36: What is ID3 approach?**  
**Ans: **The ID3 algorithm is a popular algorithm used to construct decision trees for classification tasks. It works by iteratively splitting the data based on the attribute that provides the most information gain.  
Splitting Criteria:  
• Classification: Uses Entropy or information gain to determine the best split.   

**Question 37: What do you mean by Bagging Algorithm?**  
**Ans: **Bagging, short for Bootstrap Aggregating, is a machine learning ensemble method that enhances model performance by training multiple base models on different subsets of the training data and then combining their predictions.   
**This method reduces variance and helps prevent overfitting, resulting in a more stable and accurate model. **  

**Question 38. What do you mean by Boosting Algorithm, how will it work?**  
**Ans: **Boosting algorithms build strong predictive models by combining the outputs of multiple weak learners, iteratively focusing on difficult-to-classify examples.  
Each iteration assigns weights to training instances, giving more weight to those that were misclassified in previous iterations.   
**This allows the algorithm to build a powerful model that is more accurate than any single weak learner. **  

**Question 39: How Random Forest will work?**  
**Ans: **Random Forest, a machine learning algorithm, operates by combining multiple decision trees to make predictions.   
Each tree is trained on a random subset of the data and features, ensuring diversity in the ensemble.   
**The final prediction is then made by aggregating the predictions of all trees (majority voting for classification, averaging for regression). **  

**Question 40: What is out of bag score? (OOB score)**  
**Ans: **The Out-of-Bag (OOB) score is a way to estimate the generalization error of a machine learning model without needing a separate validation dataset.   
**It's particularly useful for ensemble methods like Random Forests that use bootstrapping. **  
The OOB score is calculated by predicting the out-of-bag samples using the models that were not trained on those samples. This provides an unbiased estimate of how well the model would perform on unseen data.

**Question 41: In how many ways we can divide our data and why?**  
**Ans: **Machin Learning models, our data has been divided intoTraining data, Validation data and Testing data.  
• Training data is used for training the ML model  
• Validation data is used for hypermeter tunning  
• Testing data is used for testing our ML model.  

**Question 42: How PCA will work on?**  
Ans:   
Step 1: Standardize the Data  
Step 2: Calculate Covariance Matrix  
Step 3: Finding the Eigen Values and Eigen Vectors  
Step 4: Find the Best Principal Components  

**Question 43: What is Eigen value and Eigen vectors?**  
**Ans: **  
**Eigen Vector: **In Principal Component Analysis (PCA), eigenvectors play a crucial role in identifying the directions of maximum variance in the data.  
**Eigen Value: In Principal Component Analysis (PCA), eigenvalues are scalar values that represent the amount of variance explained by each principal component. **  

**Question 44: What is Eigen Decomposition matrix?**  
Ans: Eigen decomposition, also known as eigenvalue decomposition, is a process that breaks down a square matrix into a set of eigenvectors and eigenvalues.  

**Question 45: What is Specificity in Machine Learning?**  
**Ans: **In machine learning, specificity measures a model's ability to correctly identify negative instances (true negatives) out of all actual negative instances.  

**Question 46: What is Sensitivity in Machine Learning?**  
**Ans: **In machine learning, sensitivity, also known as recall or true positive rate, measures a model's ability to correctly identify positive instances in a dataset.  

**Question 47:  What are black box models and white box models?**  
**Ans:**  
**Black box Models**: In machine learning, a black box model refers to a model whose internal workings and decision-making processes are opaque and not easily accessible.  
Example: Deep Neural Networks, Random Forests, Boosting algorithms  
**White Box Models**:**  **White-box machine learning models are transparent and interpretable, meaning you can understand how they make predictions and the reasoning behind their decisions.  
Example: Linear Regression, Decision Trees, Rule-Based Systems  

**Question 48: What is deep copy?**  
**Ans: **In Python, a deep copy creates a new object and recursively copies all the objects found in the original, including nested objects.   
This means that the copy and the original are completely independent, and changes made to one will not affect the other. The deep copy() function from the copy module is used to perform a deep copy.  

**Question 49: What is shallow copy?**  
**Ans: **A shallow copy in Python creates a new object and then inserts references to the objects found in the original object.   
It copies the top-level structure, but not the nested objects.  

**Question 50: What is mutable in Python with examples?**  
Ans: Mutability refers to an object's ability to be modified after it's created. If an object is mutable, its value or state can be changed without creating a new object.  
This means operations can directly alter the object's contents, and these changes are reflected wherever the object is referenced.   
Example: Lists, Dictionaries, Sets, Byte arrays 

**Question 51: What is immutable with examples in Python?**  
**Ans: I**mmutability signifies that once an object is created, its value cannot be changed.   
Attempting to modify an immutable object will result in the creation of a new object instead of altering the original one.  
**Example: int, float, bool, str, tuple, frozen set**  

**Question 52: What is iterator, iterable, iteration?**  
**Ans: **  
Iterable:  
• An iterable is an object that can be looped over, meaning its elements can be accessed sequentially.  
• It has a method named __iter__() that returns an iterator.  
• Examples include lists, tuples, strings, dictionaries, and sets.  
Iterator:  
• An iterator is an object that provides a way to traverse through the elements of an iterable.  
• It implements the iterator protocol, which includes two methods:  
o __iter__(): Returns the iterator object itself.  
o __next__(): Returns the next element in the sequence. When there are no more elements, it raises a Stop Iteration exception.  
• Iterators maintain their internal state, remembering which element to return next.  
Iteration:  
• Iteration is the process of looping through the elements of an iterable using an iterator.  
• This is typically achieved with a for loop, which implicitly creates an iterator from the iterable and calls __next__() in each iteration until a Stop Iteration exception is raised.  

**Question 53: What is Generator?**  
**Ans: In Python, a generator is a function that returns an iterator object. It produces values one at a time, on demand, instead of storing the entire sequence in memory at once. **  

**Question 54: What is decorator?**  
**Ans: **In Python, a decorator is a design pattern that allows you to modify the behaviour of a function or a class without directly changing its source code.  
It's a way to "wrap" a function with additional functionality, which can be useful for tasks like logging, timing, or access control.   

**Question 55: What is list comprehension?**  
**Ans: **List comprehension in Python is a concise way to create lists.   
It provides a shorter syntax when you want to create a new list based on the values of an existing list.  

**Question 56: What is lambda function?**  
**Ans: In Python, a lambda function is a small, anonymous function defined using the lambda keyword.**  

**Question 57: What is OOPS concepts?**  
**Ans: Object-Oriented Programming (OOP) is a programming paradigm that revolves around the concept of "objects." These objects encapsulate both data (attributes) and the functions (methods) that operate on that data.**  

**Question 58: What is encapsulation?**  
**Ans: Encapsulation involves bundling data (attributes) and methods (functions) that operate on that data into a single unit, known as a class. **  

**Question 59: What is polymorphism?**  
**Ans: Polymorphism, derived from Greek, means "many forms." Which allows objects of different classes to be treated as objects of a common superclass. This enables a single interface to represent different types, enhancing code flexibility and reusability.**  

**Question 60: What is inheritance?**  
**Ans: Inheritance in Python is a mechanism that allows a class to inherit properties and methods from another class. This promotes code reusability and establishes hierarchical relationships between classes. **  
Parent class is the class being inherited from, also called base class.  
A child class inherits all the attributes and methods of its parent class also called derived class.  

**Question 61: What is method over loading?**  
**Ans: Method overloading is a feature of object-oriented programming that allows a class to have multiple methods with the same name, but with different parameters. This means that the methods can perform different tasks based on the number or type of arguments that are passed to them. **  

**Question 62: Differences between method over riding and method over loading?**  
**Ans: **  
• Overloading: involves having multiple methods with the same name but different parameters within the same class.  
• Overriding: involves replacing a method in a superclass with a new implementation in a subclass.  

**Question 63: What is public, protected and private variables?**  
**Ans: **  
1.Public:   
• Accessible from anywhere (within or outside the class).  
• Allows unrestricted access and modification of the variable or method.  
2. Protected:   
• Accessible within the class itself.  
• Accessible to subclasses (derived classes) of the class.  
• Provides limited access outside the class, primarily for inheritance purposes.  
3. Private:   
• Accessible only within the class itself.  
• Not accessible to subclasses or any other code outside the class.  
• Ensures strong encapsulation and data protection.  

**Question 64: What is statistics?**  
Ans: In data science, statistics is the branch of mathematics concerned with collecting, analysing, interpreting, and presenting data. It provides the tools and techniques to understand patterns, quantify uncertainty, and make informed decisions based on data  

**Question 65: Types of statistics?**  
Ans: In data science, statistics is broadly categorized into descriptive statistics and inferential statistics.   
Descriptive statistics summarizes and organizes data. Ex: Histograms, Bar-chart, pie-chart, violin plot, scatter plot etc  
while inferential statistics uses sample data to draw conclusions about a larger population. Ex: Z-Test, T-Test, Annova-Test, Variance, Standard Deviation, Mean, Median, Mode etc.  

**Question 66:** What is KDE plot?  
Ans: kdeplot () function is used to plot the data against a single/univariate variable. It represents the probability distribution of the data values as the area under the plotted curve.  

**Question 67: What is variable?**  
Ans: In Python, a variable is a named storage location used to hold data values. It acts as a container or a placeholder for information that can be accessed and manipulated throughout a program.  

**Question 68: What is variance?**  
Ans: Variance is a statistical measure that quantifies the spread or dispersion of a set of data points around its mean (average)  

**Question 69: What is standard deviation?**  
Ans: Standard deviation is a statistical measure that describes the amount of variation or dispersion of a set of data points around its mean (average).  

**Question 70: What is mean, median and mode?**  
Ans: Mean, median, and mode are all measures of central tendency used to describe the "average" or typical value within a dataset.  

**Question 71: Any idea about five number Summary?**  
Ans: The five-number summary is a descriptive statistical measure that provides a concise overview of a dataset's distribution using five key values.   
These values are:   
a. The minimum,   
b. The first quartile (Q1),   
c. The median (Q2),   
d. The third quartile (Q3), and   
e. The maximum  

**Question 72: What is normal/gaussian distribution?**
Ans: A normal distribution, also known as a Gaussian distribution, is a symmetrical, bell-shaped probability distribution that is fundamental in statistics.  
Where data is distributed equally on both sides (Left side and Right side)  

**Question 73: What is right skewed distribution?**  
Ans: A right-skewed distribution, also known as a positive skew, is a type of distribution where the tail on the right side is longer or fatter than the left side.  
Example: Income Distribution, Wealth distribution, House prices, Number of Children etc.  

**Question 74: What is left skewed distribution?**  
Ans: A left-skewed distribution, also known as a negatively skewed distribution, is a type of probability distribution where the tail is longer on the left side of the distribution, and the majority of the data points are clustered on the right side.  
Example: Test scores on easy exams, Retirement age etc.  

**Question 75: How do we find out the data is following Gaussian distribution or not??**  
Ans: To determine if data is Gaussian (normally distributed), you can use both visual and statistical methods. Visual methods involve plotting histograms and Q-Q plots.  

**Question 76: What is Deep Learning?**  
Ans: Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to analyse data and learn complex patterns.  
In simple words, to mimic the human brain we use deep learning.  

**Question 77: What is neuron?**  
**Ans: **In deep learning, a neuron is a fundamental computational unit within a neural network. It receives inputs, performs calculations, and produces an output, which is then passed on to other neurons in the network.  
**Think of it as a node that processes information and transmits it through connections called synapses. **  

**Question 78: What is Perceptron**?  
Ans: In deep learning, a perceptron is a fundamental building block of artificial neural networks, functioning as a simple binary classifier.  

**Question 79: What is ANN?**  
Ans: In deep learning, an Artificial Neural Network (ANN) is a computational model inspired by the structure and function of the human brain. It consists of interconnected nodes, or "neurons," organized in layers, that process and transmit information.  
In simple words, ANN is a Multi Layered Neural Network. For classification and regression tasks we will use this.  

**Question 80: What is CNN?**  
Ans: In deep learning, a Convolutional Neural Network (CNN) is a specialized type of artificial neural network primarily used for analysing visual imagery.  
Whenever our input data is images, we can use CNN.  

**Question 81: What is RNN?**  
Ans: In deep learning, a Recurrent Neural Network (RNN) is a type of neural network designed to process sequential data, where the order of elements is crucial.  
These RNN’s are well-suited for tasks involving sequences like natural language processing, speech recognition, and time series analysis. 

**Question 82: What is forward Propagation?**  
**Ans: **The process of passing the input through hidden layers by initializing some weights, adding some bias and applying activation function on top of that, we will get some outputalong with some loss.  
This entire process is called forward propagation.  
Forward propagation is the way data moves from left (input layer) to right (output layer) in the neural network.  

**Question 83: What is backward Propagation?**
Ans: To reduce the loss which is there in forward propagation, we need to update the weights.  
The process of updating the weights using optimizers is called Back Propagation or Backward Propagation.  
Backward Propagation is the process of moving from right (output layer) to left (input layer)  

**Question 84: What is activation function?**  
Ans: Which determines whether the neuron is activated or not.  

**Question 85: List of activation functions?** 
Ans:   Sigmoid: This function squashes the input into a range between 0 and 1, making it useful for output layers in binary classification tasks. However, it can suffer from the vanishing gradient problem.  
Tanh: Similar to Sigmoid, Tanh also has an S-shape, but it outputs values between -1 and 1, which is often preferred as it's zero-centred.   
ReLU (Rectified Linear Unit): ReLU outputs the input directly if it's positive, and zero otherwise. It's computationally efficient and widely used in hidden layers. A drawback is that it can suffer from the "dying ReLU" problem where neurons can become inactive  
Leaky ReLU: It is an improvement over ReLU, allowing a small, non-zero gradient for negative inputs, mitigating the dying ReLU problem.  
SoftMax: SoftMax is typically used in the output layer for multi-class classification problems. It converts the outputs into a probability distribution, where each value represents the probability of belonging to a specific class.   

**Question 86: What is Vanishing Gradient Problem?**  
Ans: It occurs when the gradients used to update the network's weights become extremely small as they propagate backward through the layers during backpropagation.  

**Question 87: What is type Casting?**  
Ans:  Type casting, also known as type conversion, is the process of changing a variable's data type into another data type  

**Question 88: What is Optimizer?**  
Ans: An optimizer is an algorithm or method used to adjust the parameters (like weights and biases) of a neural network in order to minimize a loss function.  

**Question 89: List of Optimizers?**  
• Gradient Descent  
• Stochastic Gradient Descent (SGD)  
• Mini-batch Stochastic Gradient Descent (SGD)  
• SGD with Momentum  
• Adagrad  
• RMSprop  
• Adam (Adaptive Moment Estimation) 

**Question 90: What is Loss function?**  
Ans: It’s a crucial component that quantifies the difference between a neural network's predicted output and the actual target values within a dataset  
List of loss functions?  

**Question 91: What is Chain Rule of Differentiation?**  
Ans: It is a fundamental concept used in backpropagation for training neural networks.   
It allows us to compute the gradients of the loss function with respect to each weight in the network, enabling adjustments to those weights to minimize the loss.   

**Question 92: Exploding Gradient Problem?**  
Ans: It occurs when gradients, used to update network weights during training, become excessively large during backpropagation, leading to unstable training and poor model performance  

**Question 93: Weights initialization techniques?**  
Ans: These are the methods for assigning initial values to the weights of a neural network before training.   
Proper initialization helps prevent issues like vanishing or exploding gradients, leading to faster and more stable training.   
Common techniques include   
• Random initialization,   
• Xavier/Glorot initialization,   
• He initialization  

**Question 94: Drop out Layer?**  
Ans: It is a regularization technique used in deep learning to prevent overfitting by randomly deactivating (dropping out) a fraction of neurons during training. 

**Question 95: What is Stride?**  
Ans: Particularly within convolutional neural networks (CNNs), stride refers to the number of pixels by which a convolutional filter or kernel moves across the input data (like an image) during the convolution operation.  
Essentially, it dictates how "jumpy" the filter is as it scans the input  

**Question 96: What is padding?**  
**Ans: **Padding refers to adding extra layers of data, often zeros, around the input data before applying convolutional operations.   
**This technique helps preserve spatial dimensions (like height and width in images) and prevent information loss at the edges of the input. **  

**Question 97: What is pooling?**  
Ans: Pooling is a technique used to reduce the spatial dimensions (width and height) of feature maps, effectively down sampling the input data  

**Question 98: Types of RNN?**  
**Ans: **  
• One-to-One,   
• One-to-Many,   
• Many-to-One, and  
• Many-to-Many.   
**Additionally, variations like Bidirectional RNNs, LSTMs, and GRUs offer further specialization**  

**Question 99: What is NLP?**  
Ans: Its Natural Language Processing. Whenever our input data is in the form of Text, we can use this NLP Technique.  

**Question 100: What is Tokenization?**  
Ans: Tokenization in Natural Language Processing (NLP) is the process of breaking down a sequence of text into smaller units called tokens.   
These tokens can be words, characters, or subwords, and the goal is to create manageable pieces of text that can be easily processed by NLP models.  

**Question 101: What is Stopwords?**  
Ans: Stopwords are frequently occurring words that are generally removed from text during preprocessing.   
These words, such as "the," "a," "is," and "and," are considered to have little semantic value and can be filtered out without significantly affecting the meaning of the text  

**Question 102: What is Stemming?**  
Ans: Stemming is a process in Natural Language Processing (NLP) that reduces words to their root or base form, known as the stem, by removing affixes (prefixes and suffixes).  

**Question 103: What is Lemmatization?**  
Ans: Lemmatization in NLP is a text normalization technique that reduces words to their base or dictionary form, known as the lemma, while considering the word's context and meaning  

**Question 104: What are embeddings?**  
Ans: Word embeddings in NLP are a way to represent words as numerical vectors, allowing computers to understand and process text data more effectively.   
These vectors capture semantic relationships between words, meaning that words with similar meanings will have similar vector representations.  

**Question 105: What is BOW?**  
Ans:  "Bag Of Words" (BoW) is a simple way to represent text data by counting the occurrences of words, disregarding grammar and word order.   
It's a fundamental technique for tasks like document classification and sentiment analysis  

**Question 106: What is TF-IDF?**  
Ans: TF-IDF, which stands for Term Frequency-Inverse Document Frequency, is a numerical statistic that reflects the importance of a word in a document relative to a collection of documents.  
Ex: It's a widely used technique in Natural Language Processing (NLP) for tasks like information retrieval, text mining, and text classification  

**Question 107: About N-Grams?**  
Ans: N-grams are contiguous sequences of n items (typically words) from a given text. Unigrams, bigrams, and trigrams are specific types of n-grams.   
Unigrams are single words,   
• bigrams are two-word sequences, and   
• trigrams are three-word sequences  
• Bi-Grams, Tri-Grams, N-Grams  
   
**Question 107: Central Limit Theorem?**  
Ans: The Central Limit Theorem (CLT) states that for a sufficiently large sample size, the distribution of sample means will be approximately normal, regardless of the original population's distribution.   
This fundamental statistical concept ensures that, as you collect more samples, their average values will follow a bell-shaped (normal) curve, allowing for the application of normal distribution-based statistical tests even when the underlying data is not normal.   

Word to vec, Avg word to Vec,  
Encoder and Decoders  
Transformers?  
Architecture of Transformers?  
Self-Attention Layer?  
Feed Forward Neural Network?  
Positional encodings?  
What are LLM’s?  
What is Vector Database?  
List of Vector Databases?  
List of LLM’s?  
Location Invariant?  
What is Flask application?  
What is Streamlit application?  

Example 1: Basic List of Strings  
words = ["zebra", "apple", "Banana", "123", "Zoo", "cat"]  
sorted words = sorted(words)  
print(sorted_words)  
Output:  
['123', 'Banana', 'Zoo', 'apple', 'cat', 'zebra']  
Sorting Pattern:  
It compares:  
First character of each string  
If same, then second character, and so on...  
Characters are ordered by their Unicode values:  
Lexicographical order = character-by-character comparison using Unicode values  
Digits (0-9) → ASCII 48–57  
Uppercase letters (A-Z) → ASCII 65–90  
Lowercase letters (a-z) → ASCII 97–122  
   
   
   
   
   
