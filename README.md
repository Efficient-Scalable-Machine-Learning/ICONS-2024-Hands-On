# Neuromorphic Language Modeling
In this tutorial, we will implement a language generator based on a pre-trained Event-based Gated Recurrent Unit (EGRU).
The following instructions are mere suggestions, feel free to set your own goals!
1. Run EGRU on a paragraph of text (newspaper article, wiki article...), and measure the activity sparsity
2. Implement a language generator that takes a promt (i.e. a string) as input and generates text starting from the prompt

Strech goals:
- Implement EGRU yourself in PyTorch or JAX, load the parameters from the pretrained model and check your results. How fast is the inference compared to our CUDA implementation? Can you replace the sigmoid and tanh functions with hardware-amenable alternatives? You probably have to fine-tune the model to match the quality of the pretrained model.
- Fine-tune the model on a task of your choice (Sentinent analysis, generating your favorite poems, compressing text by removing vocals, etc.)

## Installation
Tested with Python 3.10 on CPU.
Please see [Install pytorch](https://pytorch.org/get-started/locally/) in case installing from the `requirements.txt` does not work for you.
- Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
- Install jupyter notebook if you havent't yet
    ```bash
    pip install jupyter
    ```
  