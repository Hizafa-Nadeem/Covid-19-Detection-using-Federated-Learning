# Covid-19-Detection-using-Federated-Learning

Covid-19 Detection is a technique to detect Covid-19 using federated learning. Federated Learning ensures privacy and training of models on diverse dataset.
We have implemented a simple system to Detect Covid 19 using Xray images. Performed 3 experiments in different settings

Experiment 1 : Server and Clients are implemented in same jupyter notebooks
Experiment 2: Server is implemented using Django and Clients are running on a jupyter notebook.

Dataset: X-ray images dataset downloaded from Kaggle.

Models:

1. DarkNet : Trained the model from scratch. Architecture of the model is available at 
2. ConvNext: Pretrained model. The model was pretrained on imagenet. We finetuned the model on Covid-19 Dataset.


