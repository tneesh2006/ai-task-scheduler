# 🗓️ AI Task Scheduler

A smart task scheduling web application built using Python, Streamlit, and Machine Learning.

This project helps users organize their tasks based on deadlines, duration, and priority. It automatically ranks tasks by urgency and generates an optimized daily schedule to improve productivity.

The system combines heuristic scoring, K-Means clustering (unsupervised learning), and a greedy scheduling algorithm to intelligently manage workload distribution.

---

##  Key Features

-  Multi-factor Priority Scoring System
-  Urgency Detection using K-Means Clustering
-  Automatic Daily Schedule Generation
-  Workload Visualization with Charts
-  Clean and Interactive Streamlit Interface
-  Fully Deployable Web Application

---

##  How the System Works

### 1️ Priority Scoring Model

Each task is assigned a priority score using the formula:

priority_score = (5 / days_left) + (2 × priority_num) + (1 × duration)

This ensures:
- Tasks with closer deadlines are ranked higher
- Manual priority (Low, Medium, High) influences importance
- Longer tasks receive appropriate weight

This scoring mechanism helps the system make informed scheduling decisions.

---

### 2️ Machine Learning – Urgency Clustering

The application applies **K-Means clustering** to group tasks based on:

- Days Left
- Task Duration
- Computed Priority Score

Before clustering, features are scaled using StandardScaler.  
Tasks are automatically grouped into:

- 🔴 Urgent  
- 🟡 Moderate  
- 🟢 Relaxed  

This allows the system to identify workload intensity without predefined labels.

---

### 3️ Greedy Scheduling Algorithm

After ranking tasks by priority score:

- Tasks are allocated day-by-day
- Daily working hour limits are respected
- When a day becomes full, scheduling moves to the next available day

This ensures efficient and practical time management.

---

##  Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

##  Running the Project Locally

Install dependencies:

pip install -r requirements.txt  

Run the application:

python -m streamlit run app.py

---

##  Project Objective

The objective of this project is to demonstrate the practical application of AI concepts such as heuristic modeling, unsupervised learning, and optimization algorithms in solving real-world productivity problems.

---

## Author

Tanisha Jain  
B.Tech CSE Student
