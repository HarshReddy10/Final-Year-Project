# Vulnerability Detection in Blockchain Smart Contracts

This project aims to detect vulnerabilities in Ethereum smart contracts using 
graph-based analysis and machine learning models.  
It was developed as the Final Year Project for our B.Tech degree.

---

## 📌 Project Overview

Smart contracts often contain hidden vulnerabilities such as:
- Re-entrancy attacks  
- Unchecked external calls  
- Faulty withdrawal patterns  
- Improper state updates  

Our project processes Solidity contracts into graph representations (AST/CFG), 
extracts meaningful features, and uses ML models to classify the presence of vulnerabilities.

---

## 📁 Repository Structure

```
Final-Year-Project/
│
├── contracts/               # Solidity contracts (vulnerable & fixed versions)
│   ├── Attack.sol
│   ├── AttackerContract.sol
│   ├── BankWithSolution.sol
│   └── bank.sol
│
├── src/                    # Python scripts for analysis
│   ├── contract_to_graph.py
│   ├── dataset_analysis.py
│   └── newextramodel.py
│
├── data/                   # (Optional) dataset folder — add your datasets here
│
└── README.md
```




---

## 🧠 How It Works

1. **Graph Generation**  
   `contract_to_graph.py` converts Solidity contracts into structured graphs (nodes + edges).

2. **Dataset Analysis**  
   `dataset_analysis.py` analyzes the vulnerability classes, data distribution, and graph structure.

3. **Model Training**  
   `newextramodel.py` trains an ML or GNN model to detect smart contract vulnerabilities.

---

## 🚀 Running the Project

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Linux / Mac
venv\Scripts\activate          # Windows
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run scripts

```bash
python src/contract_to_graph.py
python src/dataset_analysis.py
python src/newextramodel.py
```


---

## 🔧 Tech Stack

- **Solidity**  
- **Python**
- **PyTorch / PyTorch Geometric**
- **NetworkX**
- **Machine Learning**  
- **Graph Based Representations (AST/CFG)**  

---

## 👨‍💻 Team Members

- **Harshith J** (PES2UG21CS195)  
- **Darshan Prashad S G** (PES2UG21CS907)  
- **Ranjitha S K** (PES2UG21CS423)  
- **Gowtham Sai G** (PES2UG21CS181)

---

## 📃 License
This project is open-source under the **MIT License**.

---
