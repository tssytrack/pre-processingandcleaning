# A class for data pre-processing and cleaning

## This class includes the following functionalities:
1. Instantiate the class with passing a file path or a pandas dataframe
2. Getting a summary report of the data with distinct values, missing values, and data type
3. Converting categorical data from string to categorical type
4. Perform one-hot encoding on categorical variables
5. Imputing missing values (both categorical and numerical)
6. Taking care of multicollinearity based on VIF and correlation
7. Visualizing correlation matrix for all the continuous variables with VIF greater than 5
8. count the frequency of each distinct value of the target variable
