The UC Irvine machine learning data repository hosts a famous dataset, the Pima Indians dataset, on whether a patient has diabetes originally owned by the National Institute of Diabetes and Digestive and Kidney Diseases and donated by Vincent Sigillito. You can find it at https://www.kaggle.com/uciml/pima-indians-diabetes-database/data. This data has a set of attributes of patients, and a categorical variable telling whether the patient is diabetic or not. For several attributes in this data set, a value of 0 may indicate a missing value of the variable. It has a total of 768 data-points.

Part 1-A) First, you will build a simple naive Bayes classifier to classify this data set. We will use 20% of the data for evaluation and the other 80% for training.

You should use a normal distribution to model each of the class-conditional distributions.

Report the accuracy of the classifier on the 20% evaluation data, where accuracy is the number of correct predictions as a fraction of total predictions.

Part 1-B) Next, you will adjust your code so that, for attributes 3 (Diastolic blood pressure), 4 (Triceps skin fold thickness), 6 (Body mass index), and 8 (Age), it regards a value of 0 as a missing value when estimating the class-conditional distributions, and the posterior.

Report the accuracy of the classifier on the 20% that was held out for evaluation.

Part 1-C) Last, you will have some experience with SVMLight, an off-the-shelf implementation of Support Vector Machines or SVMs. For now, you don't need to understand much about SVM's, we will explore them in more depth in the following exercises. You will install SVMLight, which you can find at http://svmlight.joachims.org, to train and evaluate an SVM to classify this data.

You should NOT substitute NA values for zeros for attributes 3, 4, 6, and 8.

Report the accuracy of the classifier on the held out 20%



