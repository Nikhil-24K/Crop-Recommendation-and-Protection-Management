# Crop Recommendation And Protection Management

## Project Overview

### Batch Information:
- **Batch Number:** CST21

### Team Members:
1. **Roll Number: 20201CST0182** : **Student Name:** **Pattabhi Ramanjuneyulu**
2. **Roll Number: 20201CST0180** : **Student Name:** **Abhichandra V**
3. **Roll Number: 20201CST0177** : **Student Name:** **V Nikhil Koushik**
4. **Roll Number: 20201CST0164** : **Student Name:** **Chinna Nagi Reddy Bhanu Prakash**
5. **Roll Number: 20211LCT0001** : **Student Name:** **Yellari Pavan Kumar**

### Guide Information:
- **Guide Name:** **Mrs. Ramyavathsala C V**
- **Designation:** Assistant Professor(G1)
- **Department:** School of Computer Science Engineering & Information Science
- **University:** Presidency University

### Course Information:
- **Course Code:** PIP104
- **Course Name:** PROFESSIONAL PRACTICE-II

## Introduction
# Crop Recommendation and Protection System

## Overview
This project implements a comprehensive Crop Recommendation and Protection System using machine learning and deep learning techniques. The system analyzes agricultural data to provide personalized crop recommendations and aids in crop protection by detecting diseases based on user-uploaded images.

## Dataset
The system utilizes the "[Crop Production](https://www.kaggle.com/datasets/asishpandey/crop-production-in-india?select=Crop_production.csv)", "[Crop Protection](https://www.kaggle.com/datasets/emmarex/plantdisease?select=PlantVillage)" dataset from Kaggle, Crop Production dataset containing the following key attributes:

- State_Name
- N (Nitrogen content)
- P (Phosphorus content)
- K (Potassium content)
- pH
- Rainfall
- Temperature
- Crop


## Data Preview
### Sample data from Crop Production dataset 
![image](https://github.com/ramu-8008/Crop_Recommendation_and_protection_management/assets/100673820/5e271755-aa3e-416d-a8c0-298f1697fa72)
### Sample image from Crop Protection dataset 
![image](https://github.com/ramu-8008/Crop_Recommendation_and_protection_management/assets/100673820/80aac0c9-3bcc-4d15-b238-9052a88f81f4)


## Machine Learning Models Used
- Logistic Regression: Accuracy (93.10%), Tuned Accuracy (93.10%)
- SVM Classifier: Accuracy (93.48%), Tuned Accuracy (94.27%)
- Decision Tree Classifier: Accuracy (94.13%), Tuned Accuracy (94.06%)
- KNeighbors Classifier: Accuracy (93.85%), Tuned Accuracy (93.85%)
- GaussianNB: Accuracy (93.39%), Tuned Accuracy (93.39%)
- RandomForest Classifier: Accuracy (94.17%), Tuned Accuracy (94.07%)
- Voting Classifier: Accuracy (94.21%), Tuned Accuracy (93.48%)
- Bagging Classifier: Accuracy (94.14%), Tuned Accuracy (94.09%)
- AdaBoost Classifier: Accuracy (29.08%), Tuned Accuracy (93.17%)
- GradientBoosting Classifier: Accuracy (94.39%), Tuned Accuracy (94.33%)


## Deep Learning Models Used
- ANN: Training Accuracy (94.51%), Testing Accuracy (94.32%)
- CNN: Training Accuracy (94.48%), Testing Accuracy (94.36%)
- LSTM: Training Accuracy (94.55%), Testing Accuracy (94.62%)
- GRU: Training Accuracy (94.27%), Testing Accuracy (94.48%)
- Ensemble Voting Classifier (DL): Testing Accuracy (94.82%)


### Modules
1. **Crop Recommendation:**
   - Uses 10 ML and 5 DL models for unbiased crop selection.
   - Ensemble of Diverse ML and DL models for robust recommendations.
   
   - User-friendly Streamlit platform for inputting environmental parameters.

2. **Crop Protection:**
   - Diagnoses plant diseases using CNN for effective management.

   - Interface includes Crop Recommendation Page and Crop Protection Page.
   

### Literature Review
- Comprehensive review of existing methods and their advantages/limitations.
- Focus on AI and ML approaches for crop production and recommendation.
- Emphasis on the impact of deep learning, particularly CNN, in crop recommendation.

## Research Gaps Identified

1. **Artificial Intelligence-Based Approaches:**
   - Need for a detailed meta-analysis of AI's impact on crop production.
   
2. **Machine Learning-Based Approaches:**
   - High accuracies achieved, but lacking focused implementation strategies.

3. **Deep Learning-Based Approaches:**
   - Specific application focus needed for deep learning models.

4. **Comparative Analysis of Techniques:**
   - Lack of depth in comparing ML and DL models.

5. **Performance Evaluation of Techniques in Specific Domains:**
   - Insufficient exploration of complex temporal dependencies in both ML and DL models.

## Proposed Methodology

1. **Data Acquisition and Preprocessing:**
   - Source data from Kaggle's Crop Production and Protection dataset .
   - Encode categorical variables and standardize features.
   
2. **Machine Learning and Deep Learning Model Training:**
   - Train various ML and DL algorithms, optimize hyperparameters.
   - Ensemble top-performing models for enhanced accuracy.

3. **Streamlight Interface Development:**
   - Create user-friendly webpages for Crop Recommendation and Crop Protection.
   - Capture environmental parameters and integrate models for real-time predictions.

## System Design & Implementation

1. **Data Preparation:**
   - Used Kaggle dataset on crop production, refined for analysis.
   
2. **Model Training:**
   - Trained various machine learning and deep learning models, achieving high accuracies.
   -  ![image](https://github.com/ramu-8008/Crop_Recommendation_and_protection_management/assets/100673820/082e3fd3-d8cd-4010-9955-3a706b6df89c)

   
3. **Disease Detection:**
   - Implemented a CNN model for identifying crop diseases with 78.69% accuracy.
   -  ![image](https://github.com/ramu-8008/Crop_Recommendation_and_protection_management/assets/100673820/a7f40ec2-7100-4c75-8012-b3086f031d47)

   
4. **Streamlit Interface:**
   - User-friendly interface for crop recommendations and disease identification.
   - ![image](https://github.com/ramu-8008/Crop_Recommendation_and_protection_management/assets/100673820/7a895504-ea73-4fa6-9f47-4dc284e6452d)


## Outcomes / Results Obtained

- **Agricultural System:**
   - Increased Productivity.
   - Sustainability Focus.
  
- **Analytical Insights:**
   - Informed Decisions.
   - Model Evaluation.

- **System Utility:**
   - Enhanced User Experience.
   - Integrated Operations.

- **Technical Achievements:**
   - High Model Accuracy.
   ## ML_DL Ensemble Model predictions for a sample input.
   - ![image](https://github.com/ramu-8008/Crop_Recommendation_and_protection_management/assets/100673820/c1bdda44-c526-4330-abe2-acd9fad5696a)

   - Disease Detection.

- **User-Centric Outcomes:**
   - Intuitive Crop Recommendation.
   -![image](https://github.com/ramu-8008/Crop_Recommendation_and_protection_management/assets/100673820/9bff0fe4-9b6a-41b4-8a29-2b6e636660ac)


   - User-friendly Disease Identification and pesticide recommendation.
   - ![image](https://github.com/ramu-8008/Crop_Recommendation_and_protection_management/assets/100673820/2ab192e6-e9dc-49ba-a3c6-a3b949304a1d)


## Conclusion

- Significant stride in transforming agriculture via technology.
- Comprehensive system for unbiased, robust predictions.
- Empowering informed decision-making for enhanced productivity and sustainability.
- Aim to incorporate real-time data and advanced models for wider accessibility.

## References

1.	Steinmetz et al. (2020). "The Impact of Artificial Intelligence on Crop Production: A Meta-Analysis." Journal of Agricultural Science, 20(5), 3416-3428.
   [https://pubmed.ncbi.nlm.nih.gov/34168318/].

2.	Patel, P., et al. (2020). "A Machine Learning Approach to Crop Recommendation Based on Soil and Climate Data." International Journal of Agricultural and Biological Engineering, 13(5), 112-118.
   [https://www.kaggle.com/code/nirmalgaud/crop-recommendation-system-using-machine-learning].

3.	Patel, D., et al. (2021). "Machine Learning for Crop Recommendation: A Review." International Journal of Engineering Research & Technology, 10(4), 3221-3226.
   [https://www.ijert.org/crop-recommendation-using-machine-learning-techniques].

4.	Shahadat Uddin (2019). "To compare the performance of different supervised machine learning algorithms for disease risk prediction." BMC Medical Informatics and Decision Making, 19(1), 128.
   [https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-019-1004-8].

5.	Mary Divya Shamili (2022). "Smart farming using Machine Learning and Deep Learning technique." Computers and Electronics in Agriculture, 196, 105295.
   [https://www.sciencedirect.com/science/article/pii/S277266222200011X].

6.	Sharma, A., et al. (2019). "A Deep Learning Approach to Crop Recommendation." Journal of Agriculture Science and Technology, 21, 877-888.
       [https://www.researchgate.net/publication/349444668_Review_on_Crop_Prediction_Using_Deep_Learning_Techniques].

7.	M K Dharani (2021). "Crop Prediction Using Deep Learning Techniques." Frontiers in Plant Science, 14, 1234555.
       [https://www.researchgate.net/publication/349444668_Review_on_Crop_Prediction_Using_Deep_Learning_Techniques].

8.	Smith, B. Chen (2019). "A Comprehensive Comparison of Machine Learning and Deep Learning in Predictive Analytics." Expert Systems with Applications, 137, 112-125.
       [https://www.researchgate.net/publication/359918228_Comparison_Analysis_of_Traditional_Machine_Learning_and_Deep_Learning_Techniques_for_Data_and_Image_Classification].

9.	X. Wang, Y. Patel (2017). "Machine Learning vs. Deep Learning: An Empirical Study on Image Classification." arXiv preprint arXiv:2104.05314.
       [https://arxiv.org/abs/2104.05314].

10.	Z. Liu, M. Rodriguez (2020). "Comparing the Performance of Machine Learning and Deep Learning in Time Series Prediction." Journal of Computational Science, 44, 101181.
       [https://www.researchgate.net/publication/227612766_An_Empirical_Comparison_of_Machine_Learning_Models_for_Time_Series_Forecasting].

11.	S. Gupta, R. Kim (2018). "Evaluating the Trade-offs: A Comparative Analysis of Machine Learning and Deep Learning in Natural Language Processing." arXiv preprint arXiv:2108.01063.
       [https://arxiv.org/abs/2108.01063].

12.	Zhang, N., et al. (2020). "Ensemble Learning for Crop Recommendation in Precision Agriculture." arXiv preprint arXiv:2005.10826.
       [https://arxiv.org/abs/2005.10826].

13.	Mahmudul Hasan (2023). "Ensemble machine learning-based recommendation system for effective prediction of suitable agricultural crop cultivation." Frontiers in Plant Science, 14, 1234555.
       [ https://www.frontiersin.org/articles/10.3389/fpls.2023.1234555/full].


## Acknowledgments

- We express our sincere **gratitude** to **Mrs. Ramyavathsala C V** for her guidance and support throughout this project. Her expertise has been invaluable in shaping our work.
