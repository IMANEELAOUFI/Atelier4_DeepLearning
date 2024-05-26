# Atelier4_DeepLearning

# Part 3: BERT
## Dataset: The dataset used for this task is from the Amazon Customer Reviews (https://nijianmo.github.io/amazon/index.html), specifically focusing on the "Fashion" category.

## Objective: The objective is to utilize the pre-trained BERT model (bert-base-uncased) to predict the overall ratings of Amazon fashion reviews.

## Steps:

- Load Data from JSON File:
  The data is loaded from a compressed JSON file and converted into a Pandas DataFrame. We filter out entries where the review text is not a string.

- Data Preparation:
  We extract the review texts and their corresponding overall ratings. The BERT tokenizer is used to tokenize the review texts, transforming them into a format suitable for input into   the BERT model.

- Create Dataset:
  We convert the tokenized inputs and labels into tensors and create a TensorDataset. This dataset is then split into training and evaluation datasets, ensuring a proportion of 80% training data and 20% evaluation data.

- DataLoader:
  We create DataLoader instances for both the training and evaluation datasets. These DataLoader instances help in efficiently batching the data during training and evaluation.

- Model Setup:
  We load the pre-trained BERT model for sequence classification with a single output label. The model is moved to the GPU if available. We also define an optimizer to adjust the model's parameters during training.

- Training and Evaluation:
  We train the model for a specified number of epochs. During each epoch, we compute the training loss and evaluation loss. The model is trained by feeding the batched data, computing the loss, and performing backpropagation to update the model's parameters. After training, we evaluate the model on the evaluation dataset to monitor its performance.

- Prediction:
  We demonstrate a prediction example by tokenizing a sample review text and passing it through the model to obtain the predicted rating. This step shows how the trained model can be used to predict the rating for new, unseen review texts.

## Conclusion:
By using the pre-trained BERT model, we can efficiently predict the overall ratings of Amazon fashion reviews. The BERT model's ability to understand the context and semantics of the text helps in making accurate predictions. This approach demonstrates the power of fine-tuning pre-trained transformer models on specific tasks to achieve high performance.
