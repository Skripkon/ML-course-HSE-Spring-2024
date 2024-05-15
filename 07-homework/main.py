from model import Model
import argparse
import warnings
import os
warnings.filterwarnings("ignore")


def split_data(data, test_size: float):
    df = data.copy()
    df = df.sample(frac=1)  # shuffle data
    test_samples = int(len(df) * test_size)
    train_samples = len(df) - test_samples
    train_data = df.iloc[0:train_samples].copy()
    test_data = df.iloc[train_samples:].copy()
    return train_data, test_data


def parse_command(mode=None, model_path=None, data=None, test=None, split=None):
    if data is None:
        return "Specify path to data with -d\n"
    if model_path is None:
        return "Specify path to model with -m\n"
    model = Model()
    test_copy = test
    match mode:
        case "train":
            if not os.path.isfile(data):
                return f"File {data} doesn't exist\n"
            if split is None:
                model.train(data=data, save_fp=model_path)
            else:
                if float(split) <= 0 or float(split) >= 1:
                    return f"Split must be in (0, 1). Split={split} was given\n"
                data = model.load_data(data)
                train, test = split_data(data=data, test_size=float(split))
                model.train(data=train, save_fp=model_path)
                predictions = model.predict(test, model_path)
                true_vals = test["rating"] > 3
                true_vals = true_vals.astype(int)
                return f"f1_score on test data={model.compute_metrics(pred=predictions, true=true_vals)}\n"
            if test_copy is not None:
                test = model.load_data(test)
                predictions = model.predict(test, model_path)
                true_vals = test["rating"] > 3
                true_vals = true_vals.astype(int)
                return f"f1_score on {test_copy}={model.compute_metrics(pred=predictions, true=true_vals)}\n"

        case "predict":
            if os.path.isfile(data):
                data = model.load_data(data)
            predictions = model.predict(data, model_path)
            return '\n'.join(str(pred) for pred in predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Reviews Classifier',
        description='This program can train a model based on CatBoost'
        'classify reviews. Then, you can use it for predictions',
        epilog='Text at the bottom of help')

    parser.add_argument('mode', choices=['train', 'predict'])
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-d', '--data', required=True)
    parser.add_argument('-t', '--test', required=False)
    parser.add_argument('-s', '--split', required=False)
    args = parser.parse_args()
    print(parse_command(mode=args.mode, model_path=args.model,
                        data=args.data, test=args.test, split=args.split))
