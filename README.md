# Risk-Prediction: Predict the risk of default payment in online purchase orders
## Task: 
The given dataset has 30,000 online purchase orders for an online trader. Each example in the data set corresponds to an online purchase order and is characterized by 44 attributes. The second attribute is the target (i.e., class) attribute that indicates whether an order has a high risk of default payment. The class attribute has two values, "yes" meaning high risk and "no" meaning low risk. 
The task is to help the online trader recognize if a person who makes an order is a customer who will eventually pay for the goods using machine learning techniques. Use a classification algorithm to build a prediction model based on the training data. This prediction model shall then classify incoming orders into the high-risk or low-risk class. The incoming orders are stored in a test data set containing 20,000 orders with an unknown risk of default payment (that is, the target attribute is missing in the test data set). 

## Data Description
Dataset Description:
ORDER_ID:  ID number of the online order\
CLASS:  Class attribute of the task (yes/no)\
B_EMAIL: was an email address submitted together with the order (yes/no)\
B_TELEFON: was a phone number submitted together with the order (yes/no)\
B_BIRTHDATE: Birth date, if submitted together with the order\
FLAG_LRIDENTISCH:  if address of delivery and invoice are identical then "yes" else "no"\
FLAG_NEWSLETTER:  was the E-Newsletter subscribed to together with the order (yes/no)\
Z_METHODE:  Selected method of payment\
Z_CARD_ART:  Selected type of card if payment by card is chosen\
Z_CARD_VALID:  Expiration date of the card in the form month.year\
Z_LAST_NAME:  is the name of the account or card holder identical with the name of the delivery address (yes/no)\
VALUE_ORDER:   Value of the order in Euro\
WEEKDAY_ORDER:  Weekday of the order\
TIME_ORDER:   Time of the order\
AMOUNT_ORDER:  Number of ordered items\
ANUMBER_01 to ANUMBER_10 : Data field for the number of an ordered item (10 columns)\
CHK_LADR:  Orders with the same delivery address within 3 days (yes/no)\
CHK_RADR: Orders with the same invoice address within 3 days (yes/no)\
CHK_K:  Orders with the same account number within 3 days (yes/no)\
CHK_CARD: Orders with the same card number within 3 days (yes/no)\
CHK_COOKIE:  Orders with the same browser cookie within 3 days (yes/no)\
CHK_IP:  Orders with the same browser IP within 3 days (yes/no)\
FAIL_LPLZ:  Zip code of the delivery address unknown (yes/no)\
FAIL_LORT: City of the delivery address unknown (yes/no)\
FAIL_LPLZORTMATCH:  Zip code and city of the delivery address do not fit together (yes/no)\
FAIL_RPLZ:  Zip code of the invoice address unknown (yes/no)\
FAIL_RORT:  City of the invoice address unknown (yes/no)\
FAIL_RPLZORTMATCH:     Zip code and city of the invoice address do not fit together (yes/no)\
SESSION_TIME:  Duration of the online session for the order in minutes\
NEUKUNDE:  is the ordering person a new customer (yes/no)\
AMOUNT_ORDER_PRE:  Total number of items of previous orders, if available\
VALUE_ORDER_PRE:    Total value of previous orders, if available\
DATE_LORDER:  Date of the last order, if available\
MAHN_AKT:             Current stage of reminder of the customer, if available\
MAHN_HOECHST:  Highest stage of reminder of the customer occurred up to now, if available\

## Data cleaning and preprocessing
**Data cleaning:** 
The columns that have more than 50% of missing values has been removed from dataset. We see many null values in 'Z_CARD_ART' columns because payment is done by check. Customers use three types of credit card: Visa, Eurocard, Amex. The null values in those rows where method type is check are fill using check and null values in those rows where method type is debit are fill using most common Eurocard. Null values in Last name column are fill by mode.

**Feature Encoding:** Majority columns are categorical and contains two values ‘yes’ and ‘no’. To covert categorical into numerical I mapped ‘yes’ as 1 and ‘no’ as 0 in all columns. I used feature encoding for three columns. For weekday order I used label encoding and for cards and payment method columns I used one hot encoding. 

**Feature Selection:** Some data visualization has been done. Many columns have too many outliers and most columns are skewed. If we remove outliers, then we loss data so for this analysis I keep outliers in data. After data analysis, feature selection techniques are used to select relevant features. There are two techniques are used mutual_info_classif and Extra tree classifiers. 13 features are selected for the model training. Feature scaling is used to normalized the features. 

**Handling Class Imbalance:** There is class imbalanced problem in ‘Class’ column. We see more no (‘low risk) than yes (‘high risk’). In order to avoid this oversampling technique is used to over sample minority class. Under sample Is done when dataset has too large but if we do that, we loss some important information. so, I preferred oversampling.

## Machine Learning Models 

There is no single machine learning algorithm that is immune to class imbalance, as the performance of any algorithm can be affected by the prevalence of certain classes in the training data. However, there are certain techniques that can be used to mitigate the impact of class imbalance on the performance of machine learning models.
Decision trees and Random Forests: Decision trees are known to be robust to class imbalance, and Random Forests, which are ensembles of decision trees, can further improve the performance on imbalanced datasets.
I used Decision trees, Random Forests, Logistic Regression, SVM and Naïve bayes but out of all the models Random Forests model performed good. 
The problem I faced when implementing machine learning models Is all models give good accuracy but they are performed well on low risks as it is a majority class, but performed worst in high risk as it is a minority class. 
For this dataset, our aim is to reduced False Negative (FN) and False Positive(FP). We focus more on precision, recall and f1 score rather than accuracy.

## The cost of misclassification
The cost of misclassification depends on the specific context and business goals of the online purchase order prediction problem. However, in general, misclassifying a low-risk order as high-risk (false positive) could lead to unnecessary cancellation or additional verification steps, which could negatively impact customer experience and increase costs. On the other hand, misclassifying a high-risk order as low risk (false negative) could result in potential fraud or financial loss, which could also negatively impact the business.
Therefore, in many cases, the cost of a false negative (misclassifying a high-risk order as low-risk) could be higher than the cost of a false positive (misclassifying a low-risk order as high-risk). The specific costs associated with misclassification should be carefully considered and balanced with other factors such as accuracy, precision, recall, and overall business goals when designing and evaluating a prediction model for online purchase orders.

For example, let's say that the cost of a false positive (classifying a low-risk order as high-risk) is $10, and the cost of a false negative (classifying a high-risk order as low-risk) is $100. In this case, misclassifying a high-risk order as low-risk (FN) would have a much higher cost than misclassifying a low-risk order as high-risk (FP).
To avoid misclassification cost, we need to focus on reducing false positives and false negatives. False positives occur when the model predicts a transaction as risky when it is not, while false negatives occur when the model predicts a transaction as not risky when it is.
To reduce the misclassification cost, we need to adjust the classification threshold of the model to minimize the number of false positives and false negatives. By minimizing these errors, we can reduce the cost of misclassification and improve the performance of the classification model.

## Conclusion:
Out of all models Random Forest model perform good. other models get good accuracy, but they are predicting well on low risk but predict worst on high risk. Random Forest is not much affected by class imbalanced. still more improvement is needed to reduce False Negative and False Positive. In this
case, It is still okay if model predict low risk as high risk but it is good if model predict high risk as low risk.
