import argparse
import joblib

from src.pipeline import clean_and_prepare, evaluate_models, load_dataset, train_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train steel industry energy prediction models.")
    parser.add_argument("--data", required=True, help="Path to steel_industry CSV file")
    parser.add_argument("--rf-out", default="random_forest_model.joblib", help="Output path for Random Forest model")
    parser.add_argument("--lr-out", default="linear_regression_model.joblib", help="Output path for Linear Regression model")
    args = parser.parse_args()

    df = load_dataset(args.data)
    df = clean_and_prepare(df)
    bundle = train_models(df)
    report = evaluate_models(bundle)

    joblib.dump(bundle.random_forest, args.rf_out)
    joblib.dump(bundle.linear_regression, args.lr_out)

    print("Training complete.")
    print("Model metrics:")
    for model_name, values in report.items():
        print(model_name, values)


if __name__ == "__main__":
    main()
