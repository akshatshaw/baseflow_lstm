# Spark 25 Project - Clustering Analysis

This project performs clustering analysis on gauge data using UMAP dimensionality reduction and K-means clustering.

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows Command Prompt:**

```bash
venv\Scripts\activate
```

**Windows PowerShell:**

```bash
venv\Scripts\Activate.ps1
```

**Git Bash:**

```bash
source venv/Scripts/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Notebooks

- Start with `data_preprocess.ipynb` to merge the data
- Then run `clustering.ipynb` for the clustering analysis

### 5. Deactivate Virtual Environment

```bash
deactivate
```

## Project Structure

- `data_preprocess.ipynb` - Data preprocessing and merging
- `clustering.ipynb` - UMAP and K-means clustering analysis
- `requirements.txt` - Python dependencies
- `static_all_gauges.csv` - Merged dataset (generated)
- `clustered_gauges.csv` - Results with cluster labels (generated)
