# Flight Delay Prediction & Trend Analysis with Databricks

## Project Overview
This project builds a data pipeline and machine learning model to predict flight delays and analyze trends using Azure Databricks. It demonstrates data processing, ML model training, and deployment workflows integrated with GitHub CI/CD.

## Technologies Used
- Azure Databricks (PySpark, MLlib)
- Delta Lake
- MLflow
- GitHub Actions (CI/CD)
- Python
- Apache Spark

## Features
- Data ingestion and preprocessing of flight and weather datasets
- Predictive model for flight delays
- Trend analysis and visualization
- Automated deployment pipeline using GitHub Actions

## Setup Instructions

### Prerequisites
- Azure Databricks workspace 
- GitHub repository for version control
- Databricks CLI installed locally
- Personal access tokens/secrets configured in GitHub repository for Databricks

### Local Setup
1. Clone the repo:
   ```bash
   git clone https://github.com/Rufinadcosta/Flight-Delay-Prediction-Trend-Analysis-with-Databricks.git
   cd Flight-Delay-Prediction-Trend-Analysis-with-Databricks
2. Install Python dependencies:

pip install -r requirements.txt

Configure Databricks CLI:

bash
Copy
Edit
databricks configure --token
Deployment
Push changes to the main branch to trigger GitHub Actions workflow

The workflow will import notebooks to your Databricks workspace automatically

Usage
Run notebooks in Databricks to process data, train models, and analyze results

View MLflow experiments for model tracking

Skills Demonstrated
Data processing with Apache Spark and PySpark

Building ML models with MLlib

Using Delta Lake for efficient storage

CI/CD pipeline management with GitHub Actions

Databricks workspace and notebook automation
