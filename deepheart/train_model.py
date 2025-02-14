# from parser import PCG
# from model import CNN
# import sys

# true_strs = {"True", "true", "t"}

# def load_and_train_model(model_path, load_pretrained):
#     pcg = PCG(model_path)

#     if load_pretrained:
#         pcg.load("/tmp")
#     else:
#         pcg.initialize_wav_data()

#     cnn = CNN(pcg, epochs=100, dropout=0.5)
#     cnn.train()

# if __name__ == '__main__':
#     data_path = sys.argv[1]

#     load_pretrained = False
#     if len(sys.argv) == 3:
#         load_pretrained = sys.argv[2] in true_strs

#     load_and_train_model(data_path, load_pretrained)


# from parser import PCG  # Ensure this import matches the file name and class name
# from model import CNN   # Ensure this import matches the file name and class name
# import sys

# # Define valid strings for "True"
# true_strs = {"True", "true", "t", "1"}


# def load_and_train_model(model_path, load_pretrained):
#     """
#     Load the dataset and train the CNN model.

#     Parameters
#     ----------
#     model_path : str
#         Path to the dataset directory.
#     load_pretrained : bool
#         Whether to load preprocessed data from disk.
#     """
#     # Initialize the PCG dataset
#     pcg = PCG(model_path)

#     if load_pretrained:
#         # Load preprocessed data
#         pcg.load("/tmp")
#     else:
#         # Process raw data and save it
#         pcg.initialize_wav_data()

#     # Initialize and train the CNN model
#     cnn = CNN(pcg, epochs=100, dropout=0.5)
#     cnn.train()


# if __name__ == '__main__':
#     # Check for correct number of arguments
#     if len(sys.argv) < 2:
#         print("Usage: python script.py <data_path> [load_pretrained]")
#         print("Example: python script.py /path/to/data true")
#         sys.exit(1)

#     # Parse command-line arguments
#     data_path = sys.argv[1]

#     # Default to False if no second argument is provided
#     load_pretrained = False
#     if len(sys.argv) == 3:
#         load_pretrained = sys.argv[2].lower() in true_strs

#     # Train the model
#     load_and_train_model(data_path, load_pretrained)

from parser import PCG
from model import CNN
import sys

# Define valid strings for "True"
true_strs = {"True", "true", "t", "1", "yes"}

def load_and_train_model(model_path, load_pretrained):
    """
    Load the dataset and train the CNN model.

    Parameters
    ----------
    model_path : str
        Path to the dataset directory.
    load_pretrained : bool
        Whether to load preprocessed data from disk.
    """
    # Initialize the PCG dataset
    pcg = PCG(model_path)

    if load_pretrained:
        # Load preprocessed data
        pcg.load("/tmp")
    else:
        # Process raw data and save it
        pcg.initialize_wav_data()

    # Initialize and train the CNN model
    cnn = CNN(pcg, epochs=100, dropout=0.5)
    cnn.train()

if __name__ == '__main__':
    # Check for correct number of arguments
    if len(sys.argv) < 2:
        print("Usage: python train.py <data_path> [load_pretrained]")
        print("Example: python train.py ./data true")
        sys.exit(1)

    # Parse command-line arguments
    data_path = sys.argv[1]

    # Default to False if no second argument is provided
    load_pretrained = False
    if len(sys.argv) == 3:
        load_pretrained = sys.argv[2].lower() in true_strs

    # Train the model
    load_and_train_model(data_path, load_pretrained)
