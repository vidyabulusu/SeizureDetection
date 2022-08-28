
# Classification of EEG Data

**_Project Objective_**

The objective of this project is to classify the Electroencephalogram(EEG) data into two classes, namely, seizures and non-seizures. The end goal is to develop a classification model that will predict seizure or non-seizure given a prescribed window of time-domain data from an EEG.

**_Data Description_**

The data contains EEG recordings collected from 500 subjects. Each subject has a file which is a continuous recording for 23.6s. Each file has 4097 data points over 23.6s. In the dataset, these 4097 points have been divided into 23 groups to get 178 data points over one second. And there are now 23 such groups of 178 per subject. Hence, the total number of records in the dataset is 23*500 = 11500.

The dataset contains patient id column, 178 columns of EEG recording data and target variable column describing the status of the patient at that second.  The following figure shows how the data has been organized in the dataset.

![](blob:https://stackedit.io/2da2c714-65d7-4ba2-96a1-44b624ff708a)

**_Assumptions_**

For this project, we will assume that the patient meta data is not influencing the classification.

The measurement units for all 178 columns are the same. EEG is typically recorded in micro-Volts (mV).

The target variable, y, is a categorical variable with values ranging from 1 to 5, describing different states of the patient during recording. Since the goal of our project is to classify as seizure or not seizure, the target variable has to be transformed into 0s for no seizure and 1s for seizure.

The dimensionality reduction will be needed to reduce 178 variables into a more manageable number.

**_“Can this project be done without Machine Learning?”_**

In the real world, a neurologist is trained to read the output of an EEG recording and find evidence of abnormal activity. However, this cannot be performed by the patients themselves or by any other monitoring devices that are currently available, based on the EEG data.



**_Data Definition_**

The dataset contains the following:

1. column_a contains hashed patient id

2. 178 columns for x variables which contain the EEG recording data

3. Last column ‘y’ which is the target variable containing values ranging from 1 to 5. Value 1 corresponds to seizure and the rest correspond to other states of the patient during the recording.


Using pandas_profiling, I was able to get the data definition for each parameter.

Parameters x1 to x178 have similar distribution.


**_Transformations required:_**

1. **_Dimensionality reduction / Transformation:_** The data recorded is in time-domain format. After researching on how to analyze EEG signal data, I learnt that the best method to analyze the data was by looking at the frequency bands. To do this, the time-domain data has to be transformed into frequency-domain by using Fast Fourier Transforms (FFT). The goal is to transform data from 178 columns into the 5 frequency bands (Hz), namely, delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–12 Hz), beta (12–30 Hz), and gamma (30–100 Hz) (see [https://en.wikipedia.org/wiki/Electroencephalography](https://en.wikipedia.org/wiki/Electroencephalography)).

2. To compute the different band powers between different frequencies, I wrote a function bandpower (inspired by open-source code). This computes area under each band by integration to calculate total power. I used periodogram and trapezoid functions from scipy library to accomplish these tasks. Once these transformations are completed, the 178 ‘x’ variables in the dataset will be transformed into 5 frequency bands.

3. The first column, column_a cannot be used because it has all unique values due to the hashing of the patient_id. Hence, that column has been dropped from the dataframe.

4. The target variable needs to be transformed. It is a categorical variable with values ranging from 1 to 5, where 1 represents a seizure and the others represent non-seizure states. The transformation will change column ‘y’ to have values of 0s and 1s where 1 represents seizure and 0 represents non-seizure.

5. After the transformation, the target, y, has 80% 0s and 20% 1s. To balance the data, I plan on using stratified split when splitting into train and test sets.

6. The data will be scaled using StandardScaler.

Please refer to the attached Jupyter notebook for details.

**_Model selection:_**

The objective of my project is to classify the EEG data, I have selected the following classification models to train:

1. Logistic Regression

2. Support Vector Machine

3. K-Means (unsupervised)

4. Random Forest

**_Dimensionality Reduction / Transformation:_**

The data recorded is in time-domain format. After researching on how to analyze EEG signal data, I learnt that the best method to analyze the data was by looking at the frequency bands. To do this, the time-domain data has to be transformed into frequency-domain by using Fast Fourier Transforms (FFT). The goal is to transform data from 178 columns into the 5 frequency bands (Hz), namely, delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–12 Hz), beta (12–30 Hz), and gamma (30–100 Hz) (see https://en.wikipedia.org/wiki/Electroencephalography).

To compute the different band powers between different frequencies, I wrote a function bandpower (inspired by open-source code). This computes area under each band by integration to calculate total power. I used periodogram and trapezoid functions from scipy library to accomplish these tasks. Once these transformations are completed, the 178 ‘x’ variables in the dataset will be transformed into 5 frequency bands

**_Justification for Dimensionality Reduction:_**

After the initial selection of the 4 models to work with and calculating training and validation accuracies and area under the curve for each model, I wanted to check the results of the same using the entire dataset with no pre-processing.

Here are my observations:

1.  In general, the Area under Curve has increased with dimensionality reduction.

2. The CPU time has significantly reduced with dimensionality reduction (about 92 times faster)

Across the board, in general, the models did better with dimensionality reduction.

**_Cross-validation: [updated]_**

Initially, the function cross_validate was not reflecting the model performance per fold. It has been updated based on the sklearn documentation, to better represent the scores from each fold and to highlight the average score per model. The supporting visualizations are in the Jupyter Notebook.

Another function cross_validate_km has been created to better represent the K-Means algorithm, since the metrics for K-Means are different from the others.

**_Hyperparameters:_**

The hyperparameters used in each model and the justification for dimensionality reduction have been listed in the Jupyter Notebook which has been submitted separately.

**_Visualizations:_**

The following visualizations are included in the Jupyter Notebook:

1. Histograms for each column in the “Feature extracted” dataset

2. Heatmap: correlation with target

3. Class distribution for target variable: histogram and scatterplots

4. Confusion matrix plots for each model

5. ROC curve for each model

**_References:_**

1. [https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay)

2. [https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html) [updated]

**_Selected Model:_**

Based on the results of cross validation for each model, Random Forest model performed the best. A comparison of ROC plots, AUC and confusion matrix for each model is presented in the Jupyter notebook. Random forest model performed the best in all three metrics and was chosen for further hyperparameter optimization.

**_Fine tuning the hyperparameters:_**

Sklearn provides multiple hyperparameters for Random Forest implementation.

The hyperparameters and their values for the Random Forest model that I used are:

· max_depth_options = [2, 5, 10, 20, 50]

· # random_state_options = [5, 10, 15, 20, 30, 40, 50]

· n_estimators_options = [5, 10, 15, 20]

· min_samples_split = [2, 5, 10]

· min_samples_leaf = [1, 2, 4]

· bootstrap = [True, False]

These parameters represent a significant decision space that needs to be explored.

I then used 2 types of fine-tuning methods to explore this space from sklearn and itertools libraries:

1. Itertools.product to create all possible combinations for the hyperparameter values chosen

2. RandomizedGridSearchCV

The itertools approach is expected to be exhaustive but slow and inefficient, however for the purpose of this study, I chose to use this approach to validate the RandomizedGridSerachCV results.

Both methods resulted in similar best parameters to be used, which were:

· 'n_estimators': 20, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 10

· RandomizedSearchCV returned 'bootstrap': True to be the best whereas itertools method found that ‘bootstrap’:False had the best score.

Details of the results obtained are tabulated and sorted in the Jupyter Notebook.

Overall, by fine tuning the hyperparameters we were able to increase the accuracy by %% and AUC by %%. It should be noted that while the model was already good, tuning resulted in a decrease of false positives and false negatives and resulted in a near perfect model. We would need to validate that this is not due to overfitting using the test data described in the next section.

**_Testing the model and Results:_**

I used the best parameters found from the above section to test the model and calculated the score and AUC metrics.

The best model produced excellent test results and confirmed that the model is indeed good performing and is not overfitting the data.

The results along with the confusion matrix are plotted in the Jupyter Notebook.

**_Future work on the project:_**

· 23 cases were predicted to be non-seizures but turned out to be seizures. While this seems small compared to the overall data, in practice if implemented in real world this could reduce the confidence of users in the model. I want to work on reducing that number

· During the dimensionality reduction exercise, I performed an FFT on the time signal to compute the band power. Then chose the bands based on existing literature. There is a potential to refine the bands further by evaluating the effect of adjusting the band widths or furthermore, use an unsupervised learning algorithm to cluster the bands based on the outcome needed. Effect of band overlap should also be investigated.

· There are a few public universities that have published raw EEG data. It would be interesting to apply this algorithm on raw EEG data to predict the results and compare the accuracy.

· It would be interesting to apply Recurrent Neural Networks and see how they perform on this data
