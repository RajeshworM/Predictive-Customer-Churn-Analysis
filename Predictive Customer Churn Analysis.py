#!/usr/bin/env python
# coding: utf-8

# <font color='Blue'> <font size='6'> **Predictive Customer Churn Analysis**
# 
# 
# **Summary of Steps** 
# 
# 1. Load the data and visualize churn distribution.*
# 
# 2. *Data Cleaning: Handle missing values, remove duplicates, and encode categorical data.*
# 
# 3. *Feature Engineering: Create new features like tenure groups and visualize their impact on churn.*
# 
# 4. *EDA: Visualize key data distributions and correlations.*
# 
# 5. *Data Preparation: Encode categorical variables, split data, and scale features.*
# 
# 6. *Model 1: Logistic Regression: Train and evaluate the Logistic Regression model and visualize the confusion matrix.*
# 
# 7. *Model 2: Decision Tree Classifier: Train and evaluate the Decision Tree model with visualization.*
# 
# 8. *Model 3: Random Forest Classifier: Train and evaluate the Random Forest model with visualization.*
# 
# 9. *Model Comparison and Insights: Compare model performance visually and extract insights.*
# 
# - This approach allows for a comprehensive analysis of the customer churn dataset, combining machine learning and visualizations to generate meaningful insights.
# 
# 
# - Important video for Format
# _(https://www.youtube.com/watch?v=jkm0s3VUjzA)_

# <font color='Blue'>**1. Load and Explore the Dataset**
# 
# *Objective: Load the dataset and inspect its structure.*

# In[25]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('customer_churn_dataset.csv')

# Check the first few rows
print(df.head())

# Get basic info about the dataset
print(df.info())

# Summary statistics
print(df.describe())

# Visualize the distribution of the 'Churn' variable
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()


# <font color='Blue'>2. **Data Cleaning: Objective**  
#   
# - Handle missing values 
# - remove duplicates
# - encode categorical data

# In[26]:


# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

# Verify changes
print(df.info())


# In[39]:


print(df.columns)


# <font color='Blue'>3. **Feature Engineering**
#     
# - Objective: Create new features or modify existing ones to improve model performance.

# In[28]:


# Example: Create tenure groups for analysis
df['Tenure_Group'] = pd.cut(df['Tenure'], bins=[0, 12, 24, 48, 60], labels=['0-12', '12-24', '24-48', '48-60'])

# Visualize Churn distribution by Tenure Group
plt.figure(figsize=(10, 6))
sns.countplot(x='Tenure_Group', hue='Churn', data=df)
plt.title('Churn by Tenure Group')
plt.show()

# Check the newly created feature
print(df[['Tenure', 'Tenure_Group']].head())


# *The plot titled "Churn by Tenure Group" shows the distribution of customer churn across different tenure groups. The x-axis represents the tenure groups (0-12, 12-24, 24-48, 48-60), and the y-axis represents the count of customers in each group. The bars indicate the number of customers who churned (1) and those who did not churn (0) within each tenure group.*

# <font color='Blue'>**4. Exploratory Data Analysis (EDA)**
# 
# - *Objective: Analyze data distributions and correlations.*

# In[31]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your DataFrame

# Filter out numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Correlation matrix to find relationships between variables
corr = numeric_df.corr()

# Plot the correlation matrix for key features
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# **The correlation matrix highlights the following important variables in relation to customer churn:**
# 
# *payment delay: This variable has the strongest positive correlation with churn, suggesting that customers who frequently delay payments are more likely to churn.*
# 
# *support calls: A moderate positive correlation indicates that customers who require frequent support calls may be more likely to churn.*
# 
# *tenure: A moderate negative correlation suggests that customers with longer tenures are less likely to churn.*
# 
# *usage frequency: A weak negative correlation suggests that customers with higher usage frequency may be less likely to churn.*
# 
# *Other variables, such as customerid, age, total spend, and last interaction, appear to have minimal or no correlation with churn. *
# 
# *However, it's important to consider other factors beyond correlation, such as domain knowledge and feature importance, when making decisions about variable selection.*
# 

# In[42]:


# Drop columns 'column2' and 'column3'
df_1 = df.drop(['customerid', 'total spend', 'last interaction'], axis=1)


# <font color='Blue'>**5. Data Preparation for Machine Learning**
#     
#  - *Objective: Prepare the data by encoding categorical features, splitting into training and test sets, and scaling.*

# In[43]:


# Convert categorical features to numerical using one-hot encoding
df_encoded = pd.get_dummies(df_1, drop_first=True)

# Separate the features (X) and the target (y)
X = df_encoded.drop(columns=['churn'])
y = df_encoded['churn']

# Train-test split (80% training, 20% testing)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (only for models that need it, like Logistic Regression)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# **6. Model 1: Logistic Regression**
# 
# Objective: Train and evaluate a Logistic Regression model.

# In[44]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Initialize and train the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_log = log_reg.predict(X_test_scaled)

# Evaluate the model
print("Logistic Regression Results")
print(classification_report(y_test, y_pred_log))
print("Accuracy:", accuracy_score(y_test, y_pred_log))

# Confusion Matrix
cm_log = confusion_matrix(y_test, y_pred_log)
plt.figure(figsize=(6,4))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()


# **7. Model 2: Decision Tree Classifier**
# 
# Objective: Train and evaluate a Decision Tree Classifier.

# In[45]:


from sklearn.tree import DecisionTreeClassifier

# Initialize and train the Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Predict on the test set
y_pred_tree = decision_tree.predict(X_test)

# Evaluate the model
print("Decision Tree Results")
print(classification_report(y_test, y_pred_tree))
print("Accuracy:", accuracy_score(y_test, y_pred_tree))

# Confusion Matrix
cm_tree = confusion_matrix(y_test, y_pred_tree)
plt.figure(figsize=(6,4))
sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Decision Tree')
plt.show()


# **8. Model 3: Random Forest Classifier**
# 
# Objective: Train and evaluate a Random Forest Classifier.

# In[46]:


from sklearn.ensemble import RandomForestClassifier

# Initialize and train the Random Forest model
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = random_forest.predict(X_test)

# Evaluate the model
print("Random Forest Results")
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6,4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.show()


# 9. Model Comparison and Insights
# Objective: Compare model performance and extract insights.

# In[47]:


# Print accuracy for each model
log_acc = accuracy_score(y_test, y_pred_log)
tree_acc = accuracy_score(y_test, y_pred_tree)
rf_acc = accuracy_score(y_test, y_pred_rf)

print(f"Logistic Regression Accuracy: {log_acc:.2f}")
print(f"Decision Tree Accuracy: {tree_acc:.2f}")
print(f"Random Forest Accuracy: {rf_acc:.2f}")

# Visualizing accuracy comparison
plt.figure(figsize=(6, 4))
models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accuracies = [log_acc, tree_acc, rf_acc]
sns.barplot(x=models, y=accuracies)
plt.title('Model Accuracy Comparison')
plt.show()


# <font color='blue'> <font size ='6'> **Conclusion**
# 
# **In the realm of customer retention, accurately predicting churn is paramount. Machine learning models offer powerful tools for this task. Among the contenders, Logistic Regression, Decision Trees, and Random Forests were evaluated for their ability to anticipate customer attrition.**
# 
# **Logistic Regression, while a valuable tool, achieved an accuracy of 0.83. Decision Trees and Random Forests, however, demonstrated exceptional performance. Decision Trees, renowned for their ability to capture complex decision-making patterns, achieved an accuracy of 0.96. Random Forests, an ensemble of Decision Trees, further enhanced predictive power, reaching an accuracy of 0.97.**
# 
# **While Random Forests emerged as the most accurate, the optimal choice of model depends on the specific characteristics of the data and the desired outcomes. For instance, if interpretability is a priority, Decision Trees may be more suitable due to their transparent representation of the decision-making process. On the other hand, if the primary goal is to achieve the highest possible accuracy, Random Forests could be the preferred choice.**
# 
# **In conclusion, machine learning models like Decision Trees and Random Forests offer significant advantages in churn prediction. By carefully considering the specific requirements of the analysis, businesses can select the most appropriate model to optimize customer retention strategies and mitigate the financial impact of customer attrition.**

# In[ ]:




