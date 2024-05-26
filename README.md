# Atelier4_DeepLearning

# Part 1: Classification Regression:


# Part 2: Transformer (Text generation):

# Dataset
The dataset used for this project is a collection of song lyrics. You can find the dataset at [https://www.kaggle.com/datasets/deepshah16/song-lyrics-dataset].

# Objective
The objective of this project is to train a GPT-2 model on song lyrics data and use the trained model to generate new song lyrics.

# Steps
- Initialization: Initialize the GPT-2 tokenizer and model, set up logging and warnings, and determine the computation device (CPU or GPU).

- Dataset: Define a custom dataset class (LyricsDataset) to load song lyrics from a CSV file (song_lyrics.csv) and preprocess them for training.

- Training Parameters: Define parameters such as batch size, number of epochs, learning rate, warmup steps, and maximum sequence length.

- Optimizer and Scheduler: Set up the AdamW optimizer and a linear scheduler for controlling the learning rate during training.

- Training Loop: Iterate over the dataset for a specified number of epochs, encode lyrics into tensors, and train the model using a forward-backward pass. Handle sequence length limitations and save the model's state after each epoch.

- Model Loading: After training, load a specific model checkpoint (gpt2_medium_lyrics_{MODEL_EPOCH}.pt) for generating new lyrics.

- Lyrics Generation: Using the loaded model, generate new song lyrics based on a starting token ("SONG:") and save the generated lyrics to a file (generated_{MODEL_EPOCH}.lyrics).

# Conclusion
This project demonstrates how to train a GPT-2 model on song lyrics data and use it to generate new song lyrics. By adjusting the dataset and training parameters, you can adapt the model to different text generation tasks.

# Part 3: BERT:

## Dataset: 
The dataset used for this task is from the Amazon Customer Reviews (https://nijianmo.github.io/amazon/index.html), specifically focusing on the "Fashion" category.

## Objective: 
The objective is to utilize the pre-trained BERT model (bert-base-uncased) to predict the overall ratings of Amazon fashion reviews.

## Steps:

- Load Data from JSON File:
I loaded the data from a compressed JSON file and converted it into a Pandas DataFrame. I filtered out entries where the review text is not a string.

- Data Preparation:
I extracted the review texts and their corresponding overall ratings. The BERT tokenizer was used to tokenize the review texts, transforming them into a format suitable for input into the BERT model.

- Create Dataset:
I converted the tokenized inputs and labels into tensors and created a TensorDataset. This dataset was then split into training and evaluation datasets, ensuring a proportion of 80% training data and 20% evaluation data.

- DataLoader:
I created DataLoader instances for both the training and evaluation datasets. These DataLoader instances help in efficiently batching the data during training and evaluation.

- Model Setup:
I loaded the pre-trained BERT model for sequence classification with a single output label. The model was moved to the GPU if available. I also defined an optimizer to adjust the model's parameters during training.

- Training and Evaluation:
I trained the model for a specified number of epochs. During each epoch, I computed the training loss and evaluation loss. The model was trained by feeding the batched data, computing the loss, and performing backpropagation to update the model's parameters. After training, I evaluated the model on the evaluation dataset to monitor its performance.

- Prediction:
I demonstrated a prediction example by tokenizing a sample review text and passing it through the model to obtain the predicted rating. This step shows how the trained model can be used to predict the rating for new, unseen review texts.

## Conclusion:
By using the pre-trained BERT model, I can efficiently predict the overall ratings of Amazon fashion reviews. The BERT model's ability to understand the context and semantics of the text helps in making accurate predictions. This approach demonstrates the power of fine-tuning pre-trained transformer models on specific tasks to achieve high performance.
