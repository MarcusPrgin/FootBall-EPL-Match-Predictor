# Predicting EPL Match Winners with Machine Learning

This project uses **machine learning** to predict the winner of English Premier League (EPL) football matches using historical match data.

The workflow follows a full end-to-end data science pipeline:
- Cleaning scraped football match data
- Engineering useful predictors
- Training a machine learning model
- Evaluating performance
- Improving predictions with rolling averages

This project is based on a tutorial by Dataquest and is suitable for beginners who want hands-on experience applying machine learning to real sports data.

---

## Project Overview

In this project, we:
1. Load EPL match data into a pandas DataFrame
2. Investigate and handle missing data
3. Clean and prepare the data for machine learning
4. Create meaningful predictors
5. Train an initial machine learning model
6. Measure prediction error and precision
7. Improve model performance using rolling averages
8. Retrain the model with improved features
9. Combine home and away team predictions
10. Review results and outline next steps

---

## Dataset

The dataset contains historical English Premier League match data, including:
- Match date
- Home and away teams
- Match results
- Goals scored
- Match statistics used to create predictors

You **do not need to scrape the data yourself** — it can be downloaded directly from the original repository.

🔗 **Data & original code**:  
https://github.com/dataquestio/project-walkthroughs

---

## Technologies Used

- Python 3
- pandas
- scikit-learn
- Jupyter Notebook (recommended)

---

## Machine Learning Approach

- **Model**: Classification model to predict match outcome
- **Target**: Match result (win/loss)
- **Features**:
  - Team performance statistics
  - Rolling averages of past match performance
- **Evaluation Metric**:
  - Precision (focused on predicting wins correctly)

Rolling averages are used to capture recent team form and improve prediction accuracy.

## Out Put 

- The output was up dated in the newest branch to print the standard epl table 
- The Epl table consists of gp, wins, loss, draw, and points

