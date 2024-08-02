# Tutorial on Event-based State-Space Models
The core motivation for this work was the irregular time-series modeling problem presented in the paper [Simplified State Space Layers for Sequence Modeling
](https://arxiv.org/abs/2208.04933). 
We acknowledge the awesome [S5 project](https://github.com/lindermanlab/S5) and the trainer class provided by this [UvA tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide4/Research_Projects_with_JAX.html), which highly influenced our code.

Our project treats a quite general machine learning problem:
Modeling **long sequences** that are **irregularly** sampled by a possibly large number of **asynchronous** sensors.
This problem is particularly present in the field of neuromorphic computing, where event-based sensors emit up to millions events per second from asynchronous channels.

## Tutorials
To get started with event-based state-space models, we created tutorials for training and inference.
- `event_ssm_inference.ipynb` shows how to load a trained model and run inference. The models are available for download from the provided [download link](https://datashare.tu-dresden.de/s/g2dQCi792B8DqnC).
- `event_ssm_online_inference.ipynb` runs event-by-event inference with batch size one (online inference) on the DVS128 Gestures dataset and measures the throughput of the model.
- `event_ssm_train.ipynb` shows how to train a model on a reduced version of the Spiking Heidelberg Digits with just two classes. The model converges after few minutes on CPUs.

## Installation
Please clone the [Event-SSM Repository](https://github.com/Efficient-Scalable-Machine-Learning/event-ssm) and follow the installation instructions of the official.

Additional packages needed:
- `pip install -r requirements.txt`

We tested with JAX versions between `0.4.20` and `0.4.29`.
Different CUDA and JAX versions might result in slightly different results.

## Trained models
We provide our best models for [download](https://datashare.tu-dresden.de/s/g2dQCi792B8DqnC).
Check out the `tutorial_inference.ipynb` notebook to see how to load and run inference with these models.
We also provide a script to evaluate the models on the test set
```bash
python run_evaluation.py task=spiking-speech-commands checkpoint=downloaded/model/SSC
```

## Help and support
We are eager to help you with any questions or issues you might have. 
Please use the GitHub issue tracker of the [Event-SSM Repository](https://github.com/Efficient-Scalable-Machine-Learning/event-ssm).

## Citation
Please use the following when citing our work:
```
@misc{Schoene2024,
      title={Scalable Event-by-event Processing of Neuromorphic Sensory Signals With Deep State-Space Models}, 
      author={Mark Schöne and Neeraj Mohan Sushma and Jingyue Zhuge and Christian Mayr and Anand Subramoney and David Kappel},
      year={2024},
      eprint={2404.18508},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
