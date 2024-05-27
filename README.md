# Diabetics-prediction-by-using-machine-learning-<br>
#Overview<br>
Diabetes is a chronic disease with the potential to cause a worldwide health care crisis. According to International Diabetes Federation 382 million people are living with diabetes across the whole world. By 2035, this will be doubled as 592 million. Diabetes is a disease caused due to the increase level of blood glucose. This high blood glucose produces the symptoms of frequent urination, increased thirst, and increased hunger. Diabetes is a one of the leading causes of blindness, kidney failure, amputations, heart failure and stroke. When we eat, our body turns food into sugars, or glucose. At that point, our pancreas is supposed to release insulin. Insulin serves as a key to open our cells, to allow the glucose to enter and allow us to use the glucose for energy. But with diabetes, this system does not work. Type 1 and type 2 diabetes are the most common forms of the disease, but there are also other kinds, such as gestational diabetes, which occurs during pregnancy, as well as other forms. Machine learning is an emerging scientific field in data science dealing with the ways in which machines learn from experience. The aim of this project is to develop a system which can perform early prediction of diabetes for a patient with a higher accuracy by combining the results of different machine learning techniques. The algorithms like K nearest neighbour, Logistic Regression, Random forest, Support vector machine and Decision tree are used. The accuracy of the model using each of the algorithms is calculated. Then the one with a good accuracy is taken as the model for predicting the diabetes. <br>
##from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve<br>
pip install mlxtend<br>
pip install missingno<br>
pip install xgboost<br>
import numpy as np<br>
import pandas as pd<br>
import matplotlib.pyplot as plt<br>
import seaborn as sns<br>
sns.set()<br>
from mlxtend.plotting import plot_decision_regions<br>
import missingno as msno<br>
from pandas.plotting import scatter_matrix<br>
from sklearn.preprocessing import StandardScaler<br>
from sklearn.model_selection import train_test_split<br>
from sklearn import svm<br>
from sklearn import metrics<br>
import sklearn.model_selection as mod<br>
import sklearn.neighbors as nei<br>
from xgboost import XGBClassifier<br>
from sklearn.neighbors import KNeighborsClassifier<br>
from sklearn.ensemble import RandomForestClassifier<br>
from sklearn.model_selection import KFold<br>
from sklearn.model_selection import StratifiedKFold<br>
from sklearn.model_selection import LeaveOneOut<br>
from sklearn import metrics<br>
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report, precision_recall_curve, average_precision_score<br>
from sklearn.naive_bayes import MultinomialNB<br>
from sklearn.model_selection import KFold<br>
from sklearn.model_selection import cross_val_score<br>
from sklearn.feature_selection import RFECV<br>
from sklearn.svm import SVC<br>
import matplotlib.pyplot as plt<br>
import statsmodels.api as sm<br>
from sklearn.linear_model import LogisticRegression<br>
from sklearn.tree import DecisionTreeClassifier<br>
from sklearn.metrics import roc_curve<br>
import operator<br>
import warnings<br>
warnings.filte<br>
warnings('ignore')<br>
%matplotlib inline<br>
df= pd.read_csv(r"C:\Users\belea\Downloads\diabetes.csv");df<br>
df.head()<br>
df.tail()<br>
# Exploratory data asnalysis<br>
#number of rowsand coloumns in this dataset<br>
df.shape<br>
df.dtypes<br>
#information about the dataset<br>
df.info()<br>
#Statistical measures of the data<br>
df.describe()<br>
#Feature and Target<br>
df.columns<br>
#Duplicate values<br>
df.duplicated().sum()<br>
#Missing values<br>
df.isnull().sum()<br>
#Imputation<br>
df_copy =df.copy(deep=True)<br>
df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)<br>
df_copy.isnull().sum()<br>
df['Outcome'].value_counts()<br>
plt.figure(figsize=(12,6))<br>
sns.countplot(x='Outcome',data=df)<br>
plt.show()<br>
#0->Non-Diabetic<br>
#1->Diabetic<br>
#Imbalanced Dataset<br>
df.groupby('Outcome').mean()<br>
#Replace NaN Values<br>
df1=df_copy<br>
# imputing the mean value of the column to each missing value of that particular column.<br>

df1['Glucose'].fillna(df1['Glucose'].mean(),inplace=True)<br>

df1['BloodPressure'].fillna(df1['BloodPressure'].mean(),inplace=True)<br>

df1['SkinThickness'].fillna(df1['SkinThickness'].median(),inplace=True)<br>

df1['Insulin'].fillna(df1['Insulin'].median(),inplace=True)<br>

df1['BMI'].fillna(df1['BMI'].median(),inplace=True)<br>
df1.isnull().sum()<br>
#Plotting Null Count Analysis Plot<br>
msno.bar(df1)<br>
#Univariate EDA<br>
#AGE<br>
age=sns.FacetGrid(df1,col='Outcome')<br>
age.map(plt.hist,'Age')<br>
#Pregenecies<br>
df1.columns<br>
Pregnancies=sns.FacetGrid(df1,col='Outcome')<br>
Pregnancies.map(plt.hist,'Pregnancies')<br>
#diabetes Pedigree Function<br>
DiabetesPedigreeFunction = sns.FacetGrid(df1, col='Outcome')<br>
DiabetesPedigreeFunction.map(plt.hist, 'DiabetesPedigreeFunction')<br>
#Nutritional Status Based on BMI<br>
#Nutritional Status Source: World Health Organization.<br>
Nutritional_Status= pd.Series([])<br>
# Nutritional_Status = []  <br>

for i in range(len(df1)):<br>
    if df1['BMI'][i] == 0.0:<br>
        Nutritional_Status.append("NA")<br>
    elif df1['BMI'][i] < 18.5:<br>
        Nutritional_Status.append("UnderWeight")<br>
    elif df1['BMI'][i] < 25:<br>
        Nutritional_Status.append("Normal")<br>
    elif df1['BMI'][i] >= 25 and df1['BMI'][i] < 30:<br>
        Nutritional_Status.append("Overweight")<br>
    elif df1['BMI'][i] >= 30:<br>
        Nutritional_Status.append("Obese")<br>
    else:<br>
        Nutritional_Status.append(df1['BMI'][i])<br>


for i in range(len(df1)):<br>
    if df1['BMI'][i] == 0.0:<br>
        Nutritional_Status[i]="NA"<br>
    elif df1['BMI'][i] < 18.5:<br>
        Nutritional_Status[i]="Underweight"<br>
    elif df1['BMI'][i] < 25:<br>
        Nutritional_Status[i]="Normal"<br>
    elif df1['BMI'][i] >= 25 and df1['BMI'][i] < 30:<br>
        Nutritional_Status[i]="Overweight"<br>
    elif df1['BMI'][i] >= 30:<br>
        Nutritional_Status[i]="Obese"<br>
    else:<br>
        Nutritional_Status[i]= df1['BMI'][i]<br>

df1.insert(6,'Nutritional_Status',Nutritional_Status )<br>
df1.head()<br>
df1['Nutritional_Status'].value_counts()<br>
print(df1.columns)<br>
df1.columns = df1.columns.str.strip()<br>
df1['Nutritional_Status'].value_counts()<br>
df1['Nutritional_Status'].value_counts()  # Using lowercase<br>
df1['Nutritional_Status'].value_counts()  # Using uppercase<br>
NutritionalStatus=sns.FacetGrid(df1,col='Outcome')<br>
NutritionalStatus.map(plt.hist,'Nutritional_Status')<br>
OGTT = pd.Series([])<br>
for i in range(len(df1)):<br>
    if df1['Glucose'][i] == 0.0:<br>
        OGTT [i]="NA"<br>

    elif df1['Glucose'][i] <= 140:<br>
        OGTT [i]="Normal"<br>

    elif df1['Glucose'][i] > 140 and df1['Glucose'][i] <= 198:<br>
        OGTT [i]="Impaired Glucose Tolerance"<br>

    elif df1['Glucose'][i] > 198:<br>
        OGTT [i]="Diabetic Level"<br>
    else:<br>
        OGTT [i]= df1['Glucose'][i]<br>


 # Insert new column - Glucose Result<br>
df1.insert(2, "Glucose Result", OGTT)<br>
df1['Glucose Result'].value_counts()<br>
Impaired_Glucose_Tolerance_Diabetic = ((df1['Glucose'] > 140) & (df1['Glucose'] <= 198) & (df1['Outcome'] == 1)).sum()<br>
Impaired_Glucose_Tolerance_Diabetic<br>
Normal_Glucose_Diabetic = ((df1['Glucose'] != 0) & (df1['Glucose'] <= 140) & (df1['Outcome'] == 1)).sum()<br>
Normal_Glucose_Diabetic<br>
Glucose=sns.FacetGrid(df1,col='Outcome')<br>
Glucose.map(plt.hist,'Glucose')<br>
GlucoseResult=sns.FacetGrid(df1,col='Outcome')<br>
GlucoseResult.map(plt.hist,'Glucose Result')<br>
Percentile_skin_thickness = pd.Series([])<br>
women_80_or_older = df1[(df1['Age'] >= 80) & (df1['Outcome'] == 1)].shape[0]<br>
print(women_80_or_older)<br>
df1['Age'].value_counts()<br>

for i in range(len(df1)):<br>
    if df1["Age"][i] >= 20.0 and df1["Age"][i] <= 79.0:<br>
        if df1["SkinThickness"][i] == 0.0:<br>
            Percentile_skin_thickness[i]=" 0 NA"<br>
        elif df1["SkinThickness"][i] < 11.9:<br>
            Percentile_skin_thickness[i]="1 <P5th"<br>
            
        elif df1["SkinThickness"][i] == 11.9:<br>
            Percentile_skin_thickness[i]="2 P5th"<br>

        elif df1["SkinThickness"][i] > 11.9 and df1["SkinThickness"][i] < 14.0:<br>
            Percentile_skin_thickness[i]="3 P5th - P10th"<br>
            
        elif df1["SkinThickness"][i] == 14.0:<br>
            Percentile_skin_thickness[i]="4 P10th"<br>

        elif df1["SkinThickness"][i] > 14.0 and df1["SkinThickness"][i] < 15.8:<br>
            Percentile_skin_thickness[i]="5 P10th - P15th"<br>

        elif df1["SkinThickness"][i] == 15.8:<br>
            Percentile_skin_thickness[i]="6 P15th"<br>
            
        elif df1["SkinThickness"][i] > 15.8 and df1["SkinThickness"][i] < 18.0:<br>
            Percentile_skin_thickness[i]="7 P15th - P25th"<br>

        elif df1["SkinThickness"][i] == 18.0:<br>
            Percentile_skin_thickness[i]="8 P25th"<br>
            
        elif df1["SkinThickness"][i] > 18.0 and df1["SkinThickness"][i] < 23.5:<br>
            Percentile_skin_thickness[i]="9 P25th - P50th"<br>

        elif df1["SkinThickness"][i] == 23.5:<br>
            Percentile_skin_thickness[i]="10 P50th"<br>
            
        elif df1["SkinThickness"][i] > 23.5 and df1["SkinThickness"][i] < 29.0:<br>
            Percentile_skin_thickness[i]="11 P50th - P75th"<br>

        elif df1["SkinThickness"][i] == 29.0:<br>
            Percentile_skin_thickness[i]="12 P75th"<br>

        elif df1["SkinThickness"][i] > 29.0 and df1["SkinThickness"][i] < 31.9:<br>
            Percentile_skin_thickness[i]="13 P75th - P85th"<br>

        elif df1["SkinThickness"][i] == 31.9:<br>
            Percentile_skin_thickness[i]="14 P85th"<br>

        elif df1["SkinThickness"][i] > 31.9 and df1["SkinThickness"][i] < 33.7:<br>
            Percentile_skin_thickness[i]="15 P85th - P90th"<br>
            
        elif df1["SkinThickness"][i] == 33.7:<br>
            Percentile_skin_thickness[i]="16 P90th"<br>
            
        elif df1["SkinThickness"][i] > 33.7 and df1["SkinThickness"][i] < 35.9:<br>
            Percentile_skin_thickness[i]="17 P90th - P95th"<br>
            
        elif df1["SkinThickness"][i] == 35.9:<br>
            Percentile_skin_thickness[i]="18 P95th"<br>

        elif df1["SkinThickness"][i] > 35.9:<br>
            Percentile_skin_thickness[i]="19 >P95th"<br>
            
        elif df1["Age"][i] >= 80.0: #Only 1 woman is 81 years old<br>
            if df1["SkinThickness"][i] > 31.7:<br>
                Percentile_skin_thickness[i]="20 >P95th"<br>

df1.insert(4, "Percentile skin thickness", Percentile_skin_thickness)<br>
df1.head(5)<br>
# Check number of women x Percentile of skin thickness<br>
df1['Percentile skin thickness'].value_counts()<br>
diabetic_malnourished_st = ((df1['SkinThickness'] < 15.8) & (df1['Outcome'] == 1)).sum()<br>
diabetic_malnourished_st<br>
df.mean()<br>
SkinThickness=sns.FacetGrid(df1,col='Outcome')<br>
SkinThickness.map(plt.hist,'SkinThickness')<br>
#Blood Pressure<br>
df1['BloodPressure'].mean()<br>
df1['BloodPressure'].min()<br>
df1['BloodPressure'].max()<br>
BloodPressure=sns.FacetGrid(df1,col='Outcome')<br>
BloodPressure.map(plt.hist,'BloodPressure')<br>
#Insulin<br>
df1['Insulin'].mean()<br>
Insulin=sns.FacetGrid(df1,col='Outcome')<br>
Insulin.map(plt.hist,'Insulin')<br>
#BMI<br>
df1['BMI'].mean()<br>
BMI=sns.FacetGrid(df1,col='Outcome')<br>
BMI.map(plt.hist,'BMI')<br>
#Data Visualization<br>
 # Histogram<br>
df1.hist(bins=50, figsize=(20, 15))<br>
plt.show()<br>
#Distribution Plot<br>
df1.plot(kind='density', subplots=True, layout=(3,3), figsize=(20, 15), sharex=False)<br>
plt.show()<br>
df1.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(20,15))<br>
#Skew of attributes distributions<br>
skew=df1.skew(axis = 1)<br>
skew<br>
#Pairplot<br>
sns.pairplot(df1, hue='Outcome')<br>
#Correlation between all the features<br>
corr_matrix = df1.corr(method='pearson')<br>
corr_matrix<br>
plt.figure(figsize=(12,10))<br>
sns.heatmap(corr_matrix, annot = True)<br>
#Separating the Features and Target<br>
df1.columns<br>
df1.head()<br>
X = df1.drop(columns=['Outcome', 'Glucose Result', 'Percentile skin thickness', 'Nutritional_Status'])<br>
Y = df1['Outcome']<br>
X.head()<br>
Y.head()<br>
#Data Preprocessing<br>
# Standardization<br>
scaler = StandardScaler()<br>
scaler.fit(X)<br>
StandardScaler()<br>
standardized_data = scaler.transform(X)<br>
print(standardized_data)<br>
X_sc = standardized_data<br>
print(X_sc)<br>
#Train Test Split<br>
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)<br>


print(X_sc.shape, X_train.shape, X_test.shape)<br>
#Training the Model<br>
#K Neighbor Classifier<br>
knn = nei.KNeighborsClassifier(n_neighbors=5)<br>
#training the support vector Machine Classifier<br>
knn.fit(X_train, Y_train)<br>
#Accuracy Score<br>
# accuracy score on the training data<br>
X_train_prediction = knn.predict(X_train)<br>
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)<br>
print('Accuracy score of the training data : ', training_data_accuracy)<br>
# accuracy score on the test data<br>
X_test_prediction = knn.predict(X_test)<br>
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)<br>
print('Accuracy score of the test data : ', test_data_accuracy)<br>
 #Evaluation:<br>
print(confusion_matrix(Y_test,X_test_prediction ))<br>
print(classification_report(Y_test, X_test_prediction))<br>
#Support Vector Machine<br>
classifier = svm.SVC(kernel='linear')<br>
#training the support vector Machine Classifier<br>
classifier.fit(X_train, Y_train)<br>
#Model Evaluation<br>
#Accuracy Score<br>
# accuracy score on the training data<br>
X_train_prediction = classifier.predict(X_train)<br>
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)<br>
print('Accuracy score of the training data : ', training_data_accuracy)<br>
# accuracy score on the test data<br>
X_test_prediction = classifier.predict(X_test)<br>
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)<br>
print('Accuracy score of the test data : ', test_data_accuracy)<br>
 #Evaluation<br>
print(confusion_matrix(Y_test,X_test_prediction ))<br>
print(classification_report(Y_test, X_test_prediction))<br>
#Decision Tree<br>
dtree = DecisionTreeClassifier()<br>
dtree.fit(X_train, Y_train)<br>
# accuracy score on the training data<br>
X_train_prediction = dtree.predict(X_train)<br>
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)<br>
print('Accuracy score of the training data : ', training_data_accuracy)<br>
# accuracy score on the test data<br>
X_test_prediction = dtree.predict(X_test)<br>
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)<br>
print('Accuracy score of the test data : ', test_data_accuracy)<br>
#Evaluation<br>
print(confusion_matrix(Y_test,X_test_prediction ))<br>
print(classification_report(Y_test, X_test_prediction))<br>
#RandomForest<br>
rfc = RandomForestClassifier(n_estimators=200)<br>
rfc.fit(X_train, Y_train)<br>
# accuracy score on the training data<br>
X_train_prediction = rfc.predict(X_train)<br>
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)<br>
print('Accuracy score of the training data : ', training_data_accuracy)<br>
# accuracy score on the test data<br>
X_test_prediction = rfc.predict(X_test)<br>
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)<br>
print('Accuracy score of the test data : ', test_data_accuracy)<br>
#Evaluation<br>
print(confusion_matrix(Y_test,X_test_prediction ))<br>
print(classification_report(Y_test, X_test_prediction))<br>
## XgBoost<br>
xgb_model = XGBClassifier(gamma=0)<br>
xgb_model.fit(X_train, Y_train)<br>
# accuracy score on the training data<br>
X_train_prediction = xgb_model.predict(X_train)<br>
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)<br>
print('Accuracy score of the training data : ', training_data_accuracy)<br>
# accuracy score on the test data<br>
X_test_prediction = xgb_model.predict(X_test)<br>
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)<br>
print('Accuracy score of the test data : ', test_data_accuracy)<br>
print(confusion_matrix(Y_test,X_test_prediction ))<br>
print(classification_report(Y_test, X_test_prediction))<br>
#Cross Validation<br>
#Recursive feature elimination<br>
# Define KFold<br>
kf = KFold(n_splits=10, shuffle=False, random_state=None)<br>
skf = StratifiedKFold(n_splits=10, random_state=None)<br>
from sklearn.svm import SVC<br>
classifier = SVC(kernel='linear')<br>
print(dir(classifier))<br>
classifier.fit(X_train, Y_train)<br>
rfecv = RFECV (estimator=classifier,step=1, cv=skf, scoring='accuracy')<br>
rfecv.fit(X,Y)<br>
#Feature Importance<br>
# Extracting feature coefficients (only applicable for linear kernel)<br>
coefficients = classifier.coef_[0]<br>
# Creating a pandas Series with feature importances<br>
feature_importance = pd.Series(abs(coefficients), index=X_train.columns)<br>
print(feature_importance)<br>
# Extracting feature coefficients (only applicable for linear kernel)<br>
coefficients = classifier.coef_[0]<br>
# Creating a pandas Series with feature importances<br>
feature_importance = pd.Series(abs(coefficients), index=X.columns)<br>
# Plotting feature importances<br>
feature_importance.plot(kind='barh')<br>
plt.title("Feature Importances (Linear SVC)")<br>
plt.xlabel("Coefficient Magnitude")<br>
plt.ylabel("Features")<br>
plt.show()<br>
feature_names = X.columns[:10]<br>
feature_names<br>
X1 = X[feature_names]<br>
new_features = list(filter(lambda x: x[1],zip(feature_names, rfecv.support_)))<br>
new_features<br>
X_new = df1[['Pregnancies','Glucose', 'BloodPressure','SkinThickness']]<br>
scaler.fit(X_new)<br>
standardized_data = scaler.transform(X_new)<br>
Xnew_sc = standardized_data<br>
Xnew_sc<br>
from sklearn.model_selection import train_test_split<br>
X_train, X_test, Y_train, Y_test = train_test_split(Xnew_sc, Y, test_size=0.2, stratify=Y, random_state=2)<br>
print(Xnew_sc.shape, X_train.shape, X_test.shape)<br>
classifier = svm.SVC(kernel='linear',probability=True)<br>
#training the support vector Machine Classifier<br>
classifier.fit(X_train, Y_train)<br>
# accuracy score on the training data<br>
X_train_prediction = classifier.predict(X_train)<br>
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)<br>
print('Accuracy score of the test data : ', test_data_accuracy)<br>
#Evaluation<br>
print(confusion_matrix(Y_test,X_test_prediction ))<br>
print(classification_report(Y_test, X_test_prediction))<br>
#ROC Curve<br>
# Obtain predicted probabilities for the positive class (class 1)<br>
out_pred_prob = classifier.predict_proba(X_test)[:, 1]<br>
# Calculate false positive rate, true positive rate, and thresholds<br>
fpr, tpr, thresholds = roc_curve(Y_test, out_pred_prob)<br>
# Plot ROC curve<br>
plt.plot(fpr, tpr, label='ROC curve')<br>
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')<br>
plt.xlabel('False Positive Rate')<br>
plt.ylabel('True Positive Rate')<br>
plt.title('ROC Curve')<br>
plt.xlim([-0.02, 1])<br>
plt.ylim([0, 1.02])<br>
plt.legend(loc="lower right")<br>
plt.show()<br>
ras = roc_auc_score(Y_test, out_pred_prob)<br>
ras<br>
precision, recall, thresholds = precision_recall_curve(Y_test, out_pred_prob)<br>
# create plot<br>
plt.plot(precision, recall, label='Precision-recall curve')<br>
_ = plt.xlabel('Precision')<br>
_ = plt.ylabel('Recall')<br>
_ = plt.title('Precision-recall curve')<br>
_ = plt.legend(loc="lower left")<br>


aps = average_precision_score(Y_test, out_pred_prob)<br>
aps<br>


#Machine Learning Models in Diabetes Prediction<br>

#Logistic Regression<br>
Logistic regression is a widely used statistical model for binary classification problems. In the context of diabetes prediction, logistic regression models the probability that a given patient has diabetes based on input features such as age, BMI, blood pressure, and family history. Studies have demonstrated that logistic regression provides a baseline for comparison with more complex models, often achieving reasonable accuracy with interpretable results .<br>

#Decision Trees and Random Forests<br>
Decision trees classify data by recursively splitting it based on feature values. Random forests, an ensemble method that builds multiple decision trees, enhance the predictive performance and robustness against overfitting. Research shows that random forests can effectively capture complex interactions between features in diabetes datasets, leading to high predictive accuracy .<br>

#Support Vector Machines (SVM)<br>
Support Vector Machines are powerful classifiers that find the optimal hyperplane separating data points of different classes. SVMs have been applied to diabetes prediction with success, particularly in cases where the dataset has high-dimensional features. Studies have shown that SVMs, combined with kernel functions, can achieve high accuracy and generalization in predicting diabetes .<br>

#Neural Networks<br>
Artificial neural networks (ANNs), including deep learning models, have shown great promise in diabetes prediction due to their ability to learn complex patterns from large datasets. Deep neural networks (DNNs) with multiple hidden layers can capture intricate relationships between features. Convolutional neural networks (CNNs) and recurrent neural networks (RNNs) have also been explored for time-series data and image-based diabetes diagnosis .<br>

#Gradient Boosting Machines (GBM)<br>
Gradient Boosting Machines, including XGBoost and LightGBM, are ensemble learning techniques that build models in a stage-wise manner. They have been highly effective in diabetes prediction due to their ability to handle various types of data and provide robust performance. Studies have highlighted the superior accuracy and feature importance insights provided by GBM models .<br>

#Key Insights<br>

•	Data Collection and Preprocessing: Gather relevant datasets containing medical and demographic information. Clean and preprocess the data to handle missing values, normalize features, and encode categorical variables.<br>

•	Feature Selection: Identify and select the most significant features that contribute to diabetes prediction, such as glucose levels, BMI, age, and family history.<br>

•	Model Selection: Choose appropriate machine learning algorithms (e.g., logistic regression, decision trees, random forests, support vector machines) suitable for classification tasks.<br>

•	Model Training and Validation: Split the dataset into training and testing subsets. Train the chosen models on the training set and validate their performance using the testing set.<br>

•	Hyperparameter Tuning: Optimize the hyperparameters of the selected models to improve their predictive accuracy and generalization capabilities.<br>

•	Model Evaluation: Assess the models using metrics such as accuracy, precision, recall, F1 score, and ROC-AUC to determine their effectiveness in predicting diabetes.<br>

•	Deployment and Monitoring: Deploy the best-performing model into a production environment. Continuously monitor its performance and update the model as new data becomes available.<br>

•	Interpretability and Insights: Ensure the model's predictions are interpretable, providing insights into the factors contributing to diabetes risk for better clinical decision-making.<br>

#Benifits<br>
Early Detection and Diagnosis:<br>
Machine learning models can analyze large datasets to identify patterns and risk factors associated with diabetes.
Early detection allows for timely intervention, which can prevent or delay the onset of diabetes-related complications.<br>

Personalized Treatment Plans:<br>
ML models can help in creating personalized treatment plans based on an individual's unique risk factors, medical history, and lifestyle.<br>
This leads to more effective management of diabetes and better patient outcomes.<br>

Improved Accuracy:<br>
Machine learning algorithms, such as Random Forest, Support Vector Machines (SVM), and Neural Networks, can achieve high accuracy in predicting diabetes.<br>
These models can consider a multitude of variables simultaneously, often outperforming traditional statistical methods.<br>
#Dependencies
Stramlit

#Conclusion<br>
The application of machine learning in diabetes prediction provides significant advantages, from early detection and personalized treatment to improved accuracy and cost-effectiveness. By leveraging these technologies, healthcare systems can enhance patient outcomes, reduce the burden of diabetes, and foster ongoing advancements in medical research and treatment.<br>

#Acknowlagment

Streamlit: https://docs.streamlit.io/










    
    
    


