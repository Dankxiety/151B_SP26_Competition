GPU Type: Tesla T4 (via Google Colab)
Approx. total inference time on private set: 6ish hours
Model weights are on HuggingFace and are automatically loaded by `run_inference.py`
To run:
`pip install -r requirements.txt`
`python3 run_inference.py`

Note: Inference was performed on a notebook on Colab, which I tried to convert to a .py for submission. I am unable to verify the end-to-end pipeline completely works with the .py. The original notebook is in `inference_notebook.ipynb`