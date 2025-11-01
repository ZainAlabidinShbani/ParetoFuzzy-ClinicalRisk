01_data_exploration.ipynb

    Read the dataset

    General info, summary statistics

    Missing values check

    Visualize distributions (histograms, countplots, boxplots)

    Understand correlations (maybe heatmap)

    Notes on initial insights (imbalances, outliers, etc.)

02_feature_analysis.ipynb

    Data cleaning & preprocessing

    Handle missing values (bmi, maybe smoking_status)

    Convert categorical features to numeric or fuzzy categories

    Gender, Married, Work type, Residence type, Smoking

    Feature scaling / normalization for numeric inputs

    Feature selection / engineering

    Decide which inputs go to Fuzzy system

    Transform target variable stroke into Low/Medium/High Risk (if needed)

    Prepare final dataset ready for model training

    هذا النوتبوك يكون “bridge” بين EDA ومرحلة تدريب النماذج، لأنه يحول البيانات من raw إلى ready-to-use.

03_model_training.ipynb

    Fuzzy Logic baseline system

    Define Membership Functions

    Compute baseline performance

    GA-based optimization

    Encode MFs as chromosomes

    Crossover, mutation

    Evaluate improved performance

    Neural Network baseline

    Same inputs/outputs

    k-fold cross-validation

    Compute performance metrics (Accuracy, Precision, Recall, F1)

    Save trained models/results for visualization

04_results_visualization.ipynb

    Confusion matrices

    ROC curves (one-vs-all)

    Bar charts for metrics

    Compare Fuzzy baseline vs Fuzzy GA vs Neural Network

    Visualize effect of Lipschitz continuity (stability)

    Summarize final insights

✅ Key points for data cleaning / preprocessing (02_feature_analysis)

    Missing values

    bmi: replace with mean/median

    smoking_status: treat 'Unknown' as a separate category

    Categorical encoding

    Gender: Male=0, Female=1, Other=2

    Ever_married: No=0, Yes=1

    Work_type: one-hot or fuzzy categories

    Residence_type: Rural=0, Urban=1

    Smoking_status: fuzzy categories or one-hot

    Normalization

    age, avg_glucose_level, bmi → Min-Max scaling (0-1)

    Target transformation

    stroke: keep 0/1 for Neural Network

    For Fuzzy: convert to Low/Medium/High risk (based on probability or thresholds)

    Final dataset ready for modeling
