from LRCN import create_LRCN_model
from record_video import record_vid
from Helpers import predict_word_level, word_list, CLASSES_LIST, SEQUENCE_LENGTH, parse_sentence, most_freq_word, highest_prob_word

if __name__ == "__main__":

    input("Press Enter to RECORD Video...")

    file_name = "Demo.mp4"
    record_vid(file_name)

    print("\n\n Video Recorded !")

    # Creating LRCN model and loading weights
    print("\nCreating Model")
    weights_path = "Model_1-Without-CTC-LOSS_checkpoint1_tillbatch_9.h5"
    model = create_LRCN_model()
    model.load_weights(weights_path)

    print("\nModel predicting...")
    prediction = predict_word_level(file_name, word_list, model, CLASSES_LIST, SEQUENCE_LENGTH)

    #print(parse_sentence(prediction, highest_prob_word))
    print(parse_sentence(prediction, most_freq_word))


