from infer import Inference
from config import model_path, labels


def main():
    infer = Inference(
        model_path,
        labels,
    )

    infer.testInference()


if __name__ == "__main__":
    main()
