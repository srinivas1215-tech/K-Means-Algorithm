# 📊 K-Means Clustering from Scratch

## 📖 Description
This project is a **custom implementation of the K-Means Clustering Algorithm in Python**.  
Unlike libraries that provide built-in clustering, this version manually encodes, normalizes,  
and clusters data points to help understand the internal working of K-Means.  
The algorithm is tested using the **Iris dataset**.

## 🚀 Features
- ✅ Encode categorical data into numerical values
- 🔄 Normalize features for better clustering
- 📐 Custom Euclidean distance function
- 🎯 Centroid initialization and iterative updates
- 🧪 Train/Test split with accuracy calculation
- 📂 Saves predictions to `testoutput.csv`

## 🛠️ Tech Stack
- Python 🐍
- Pandas
- NumPy
- Scikit-learn (for train/test split & evaluation)
- Math library

## 📊 Example Workflow
1. Load **Iris dataset**
2. Encode + normalize features
3. Choose **K** (number of clusters)
4. Run custom K-Means
5. Evaluate with accuracy score

## ▶️ How to Run
```bash
pip install pandas numpy scikit-learn
python kmeanscluster.py
