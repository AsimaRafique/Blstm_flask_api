# from flask import Flask, request, jsonify
# import pandas as pd
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer

# app = Flask(__name__)

# # Load the BLSTM model
# model = load_model("C:\\Users\\ljk\\Downloads\\clstm_final.h5")
# tokenizer = Tokenizer()

# # Define the API endpoint for receiving text files and returning the list of action items
# @app.route('/predict', methods=['POST'])
# def predict_action_items():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})
#     data = file.read().decode('utf-8')
#     test_data = pd.DataFrame({'Sentences': [data]})
#     test_texts = test_data["Sentences"].values.astype(str)
#     max_sequence_length = 100
#     X_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=max_sequence_length)
#     test_pred = model.predict(X_test)
#     action_items = [test_texts[i] for i in range(len(test_pred)) if test_pred[i] > 0.5]
#     return jsonify({'action_items': action_items})

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer
# import pandas as pd

# app = Flask(__name__)

# # Load the tokenizer and the model
# tokenizer = Tokenizer()
# model = load_model('C:\\Users\\ljk\\Downloads\\clstm_final.h5')

# @app.route('/predict', methods=['POST'])
# def predict_action_items():
#     # Get the data from the request
#     data = request.json['data']

#     # Preprocess the data
#     sequences = tokenizer.texts_to_sequences([data])
#     padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust maxlen according to your model

#     # Make predictions
#     predictions = model.predict(padded_sequences)
#     action_items = [1 if pred > 0.5 else 0 for pred in predictions]

#     return jsonify({'action_items': action_items})

# if __name__ == '__main__':
#     app.run(debug=True)


# # lastly executed
# from flask import Flask, request, jsonify
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.models import load_model

# app = Flask(__name__)

# # Load and preprocess the model and tokenizer
# model = load_model("C:\\Users\\ljk\\Downloads\\clstm_final.h5")  # Load your saved model
# max_sequence_length = 100  # Adjust this based on your text length
# tokenizer = Tokenizer()

# @app.route('/predict', methods=['POST'])
# def predict_action_items():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             data = file.read()
#             data = data.decode('utf-8')  # Decode the data
#             lines = data.split('\n')  # Split the data into lines
#             lines = [line.strip() for line in lines if line.strip()]  # Remove empty lines
#             print(f"lines{lines}")
#             # Tokenize the text data and pad sequences
#             X_data = pad_sequences(tokenizer.texts_to_sequences(lines), maxlen=max_sequence_length)
#             print(X_data)
#             # Predict using the model
#             predictions = model.predict(X_data)
#             print(f"predictions{predictions}")
#             predicted_labels = (predictions > 0.5).astype(int)
#             print(f"predicted labels{predicted_labels}")

#             # Generate a list of action items
#             action_items = [lines[i] for i in range(len(lines)) if predicted_labels[i] == 0]

#             # Return the list of action items as a JSON response
#             return jsonify({'action_items': action_items})
#             # return jsonify(X_data)

# if __name__ == '__main__':
#     app.run(debug=True)


# # blstm final api
# from flask import Flask, request, jsonify
# from keras.models import load_model
# from keras.preprocessing import sequence
# from keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.text import Tokenizer
# import pickle
# import os


# app = Flask(__name__)

# # Load the model


# @app.route('/predict', methods=["POST"])
# def predict_action_items():
#     loaded_model = load_model(
#         "C:\\Users\\ljk\\Downloads\\my_blstm_model_final.h5")

# # Load the tokenizer from the saved file
#     with open("C:\\Users\\ljk\\Downloads\\tokenizer2_blstm.pkl", 'rb') as tokenizer_file:
#         loaded_tok = pickle.load(tokenizer_file)

#     max_len = 150  # Adjust max_len as per your training configuration

#     # Receive the text file
#     # if request.method == "POST":
#     file = request.files["file"]
#     if file:
#         # Receive the text file
#         file = request.files['file']

#         # Read the content of the file
#         content = file.read().decode('utf-8')

#         # Create an empty list to store the line-by-line data
#         lines = []
#         # Iterate over the lines in the file and add them to the list
#         for line in content:
#             lines.append(line)

#     # Tokenize and pad the input sentences
#     test_sequences = loaded_tok.texts_to_sequences([lines])
#     test_sequences_matrix = sequence.pad_sequences(
#         test_sequences, maxlen=max_len)

#     # Make predictions
#     predictions = loaded_model.predict(test_sequences_matrix)
#     print(f"predictions   {predictions}")

#     # Assuming it's a binary classification task
#     binary_predictions = (predictions > 0.5).astype(int)
#     # # Print the predictions
#     results = []
#     for sentence, prediction in zip(lines, binary_predictions):
#         print(f"Sentence: {sentence} - Predicted Label: {prediction}")
#         result = (sentence, prediction)
#         results.append(result)
#     # Convert binary predictions to a list of action items
#     # action_items = ["Action Item" if pred[0] ==
#     #                 1 else "No Action Item" for pred in binary_predictions]

#     Result = {
#         "action_items": results,
#     }

#     return jsonify(Result)

#     # except Exception as e:
#     #     return jsonify({"error": str(e)}), 500


# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
import os
from flask import Flask, request, jsonify
from moviepy.video.io.VideoFileClip import VideoFileClip
app = Flask(__name__)

# Load the model
loaded_model = load_model("my_blstm_model_final.h5")

# Load the tokenizer from the saved file
with open("tokenizer2_blstm.pkl", 'rb') as tokenizer_file:
    loaded_tok = pickle.load(tokenizer_file)

max_len = 150  # Adjust max_len as per your training configuration


@app.route('/predict', methods=["POST"])
def predict_action_items():
    try:
        # Receive the text file
        file = request.files['file']

        # Read the content of the file
        content = file.read().decode('utf-8')

        # Split the content into lines
        lines = content.splitlines()

        # Tokenize and pad each line separately
        results = []
        for line in lines:
            # Tokenize and pad the input sentence
            test_sequences = loaded_tok.texts_to_sequences([line])
            test_sequences_matrix = sequence.pad_sequences(
                test_sequences, maxlen=max_len)

            # Make predictions
            predictions = loaded_model.predict(test_sequences_matrix)

            # Assuming it's a binary classification task
            binary_predictions = (predictions > 0.5).astype(int)

            # Append results for each line
            result = {"sentence": line, "prediction": int(
                binary_predictions[0][0])}
            results.append(result)

        Result = {
            "action_items": results,
        }

        return jsonify(Result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/convert_mp4_to_wav', methods=['POST'])
def convert_mp4_to_wav():
    try:
        # Receive the MP4 file
        mp4_file = request.files['file']

        # Specify the output WAV file path
        output_wav_file = 'output.wav'

        # Save the received MP4 file
        mp4_path = 'input.mp4'
        mp4_file.save(mp4_path)

        # Convert MP4 to WAV using a context manager
        with VideoFileClip(mp4_path) as video_clip:
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(output_wav_file)

        # Clean up - delete the temporary MP4 file
        os.remove(mp4_path)

        return jsonify({"success": True, "message": "Conversion successful."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
