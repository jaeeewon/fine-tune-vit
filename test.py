from infer import Inference


def main():
    infer = Inference(
        model_path="./vit-base-beans",
        labels=["angular_leaf_spot", "bean_rust", "healthy"],
    )

    infer.testInference()


if __name__ == "__main__":
    main()
