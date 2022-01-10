from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


class ClassificationImageEngine:
    """
    Bussines object to serve model and classification.
    """
    def __init__(self, h5_model_path, labels_txt_file, input_image_size):
        assert h5_model_path, "Model path is required!"
        assert labels_txt_file, "Labels txt file path is required!"
        assert input_image_size, "Input layer size is required!"

        self.__model = load_model(h5_model_path)
        self.__labels = self.read_labels_from_txt_file(path_to_labels_txt_file=labels_txt_file)
        self.__input_img_size = input_image_size

    @staticmethod
    def read_labels_from_txt_file(path_to_labels_txt_file: str) -> list:
        """
        Method prepare and return a list from txt labels file.
        :param path_to_labels_txt_file: path to file with labels
        :return: labels list
        """
        _classes = []

        with open(path_to_labels_txt_file) as file:
            for _class in file.readlines():
                _classes.append(_class.strip())

        return _classes

    def get_label_description(self, label_id: int) -> str:
        """
        Method return a label description
        :param label_id: label id
        :return: label description
        """
        _label_desc = "Class does not exist..."

        try:
            _label_desc = self.__labels[label_id]
        except (IndexError, TypeError) as e:
            print(e)

        _label_desc = f"[{label_id}] {_label_desc}"

        return _label_desc

    def classify_from_stram(self, image_stream) -> str:
        """
        Method is used to classify a given stream image.
        :param image_stream: stream
        :return: label
        """
        img_to_predict = None

        try:
            image_input = Image.open(image_stream)
            image_input = image_input.resize((self.__input_img_size, self.__input_img_size))
            image_input = np.expand_dims(image_input, axis=0)
            image_input = image_input[..., :3]
            img_to_predict = image_input.copy()
            img_to_predict = np.asarray(img_to_predict, dtype=np.float32)

        except Exception as e:
            print(e)

        if img_to_predict is not None:
            _out = self.__model.predict(img_to_predict, steps=1)
            _label_id = np.argmax(_out, axis=1)[0]
            return self.get_label_description(_label_id)
        else:
            print("Something went wrong!!!")
            return "I can't see any object on image... sorry..."
