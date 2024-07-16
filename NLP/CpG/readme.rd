%md

To create a comprehensive README file for the project, let's break down the structure. Here's a draft based on typical elements included in a README file. You'll need to fill in specific details based on the actual code and functionality:

---

# CpG Prediction Project

This project focuses on predicting CpG sites using Natural Language Processing (NLP) techniques. The repository includes model files, Jupyter notebooks, and scripts for deploying the model using Gradio and Streamlit.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [API and Apps](#api-and-apps)

## Project Overview

The CpG Prediction Project aims to predict CpG sites, which are regions of DNA where a cytosine nucleotide is followed by a guanine nucleotide. The project leverages machine learning models trained on nucleotide sequences to perform this task.

## Directory Structure

```
CpG/
├── User_Interface_Screenshot/
├── CPG_Pred_model_padded.pth
├── CPG_count_model_padding_with_emb.pth
├── CP_Pred_model.pth
├── CP_pred_emb_model.pth
├── CpG_Api.py
├── cpg_ass.ipynb
├── cpg_emb_ass.ipynb
├── cpg_gradio_app.py
├── cpg_streamlit_app.py
```

## Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/Mahendra-V27/ml_projects_public.git
cd ml_projects_public/NLP/CpG
pip install -r requirements.txt
```

## Usage

### Jupyter Notebooks

1. **cpg_ass.ipynb**: Notebook for analyzing CpG predictions.
2. **cpg_emb_ass.ipynb**: Notebook for analyzing CpG predictions with embeddings.

Open and run these notebooks to see the analysis and model predictions.

### Running the API

Start the API by running:

```bash
python CpG_Api.py
```

### Gradio App

To launch the Gradio app:

```bash
python cpg_gradio_app.py
```

### Streamlit App

To launch the Streamlit app:

```bash
streamlit run cpg_streamlit_app
```

## Model Training

The models are pre-trained and stored as `.pth` files in the repository. To train the models from scratch, refer to the notebooks and follow the data preprocessing and training steps.

## API and Apps

- **CpG_Api.py**: REST API for CpG site prediction.
- **cpg_gradio_app.py**: Gradio-based web interface for CpG site prediction.
- **cpg_streamlit_app.py**: Streamlit-based web interface for CpG site prediction.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for review.
