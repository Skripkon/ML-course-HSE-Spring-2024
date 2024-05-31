import argparse
import warnings
import pickle
warnings.filterwarnings("ignore")

mapping = {
        1: "PYTHON",
        2: "C++",
        3: "JAVASCRIPT",
        4: "JAVA",
        5: "YAML",
        6: "BASH",
        7: "MARKDOWN",
        8: "C",
        9: "KOTLIN",
        10: "HASKELL",
        11: "PLAIN TEXT"
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Classifier',
        description='You can run model',
        epilog='no help')

    parser.add_argument('file')
    args = parser.parse_args()

    input_file = args.file
    f = open(input_file, "r")
    imput_text = f.read()
    with open("model.pkl", 'rb') as file:
        model, bow = pickle.load(file)

    x_test_python = bow.transform([imput_text])
    print(f"Model predicts {mapping[model.predict(x_test_python)[0]]}")
