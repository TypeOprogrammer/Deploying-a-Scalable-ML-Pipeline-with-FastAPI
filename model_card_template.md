# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a Random Forest Classifier designed to predict if a person's income exceeds or falls below $50,000, using demographic information from the US Census dataset located in the data folder. It utilizes features such as work class, education level, marital status, occupation, relationship status, race, sex, and country of origin. The model was built using frameworks like Scikit-learn, Python, Pandas, and NumPy. Categorical variables were one-hot encoded, and the labels were converted into binary values.



## Intended Use
The model aims to help analyze demographic income trends for research, analysis, or subsequent tasks like evaluating public policies. Data analysts can utilize it to investigate income patterns among various demographic groups, researchers can apply it to study socio-economic trends, and it can also serve as an educational tool in machine learning and data science courses. However, it is important to note that this model should not be used in decision-making processes that impact individuals as it may carry biases from the original dataset.

## Training Data
The training data is sourced from a 1994 U.S. Census dataset, containing 32,561 samples of demographic and income data. This dataset was divided into an 80% training set and a 20% testing set.



## Evaluation Data
The evaluation data is derived from the 20% of the original census dataset that was reserved for testing, amounting to roughly 6,513 rows. To ensure uniformity, the same preprocessing techniques applied to the training data were also implemented on this test data.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
Precision: 0.7417 
Recall: 0.6382
F1-Score: 0.6865

## Ethical Considerations
The U.S. Census dataset may highlight disparities in economic outcomes across demographic groups, shaped by historical inequalities, structural barriers, and limitations in the data. These factors could result in the model performing differently for certain groups, and care must be taken when interpreting the results to avoid overgeneralizing or misidentifying correlations as causations.

## Caveats and Recommendations
This model is intended for research and educational purposes and should not be used for high-stakes decision-making. The data contained in this project is over 30 years old. New data should be used for any decision making process.
