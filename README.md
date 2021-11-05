# Handwriting_CRNN
This repo details the implementation of a model that can function as an OCR for cursive handwriting. The dataset utilised is the [IAM Dataset](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database). <br>

# Architecture
The architecture 1st passes the input image into a series of Conv2D layers. The output is then reshaped to a (batch_size, time_steps, depth) shape. This is then passed through an LSTM based seq2seq encoder-decoder system (seq2seq architecture shown below).<br> 
![seq2seq](https://user-images.githubusercontent.com/36445587/140475693-ce2be61b-5bb9-4d27-8dd4-abfdf285873b.png)

Note: The text input to the decoder and outputs are tokenized and one-hot-encoded hence the loss used is categorical crossentropy rather than the CTC loss usually used for such tasks.<br>

# Usage
First Generate the labels:<br>
```bash
$ python3 label_gen.py 
```

To train the model:<br>
```bash
$ python3 main.py 
```
From line 159 in main.py onwards is the inference code which can be tested by selective running or by utilizing in another python file. 

# Working
Input image:<br>
![test](https://user-images.githubusercontent.com/36445587/140475099-5ecc2970-ad13-4144-981d-14a59069dd40.png)

Predicted output:<br>
![pred](https://user-images.githubusercontent.com/36445587/140475556-db2f7153-53a1-4d9b-8693-05ea3932c5be.JPG)
