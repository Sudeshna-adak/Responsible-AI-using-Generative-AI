import matplotlib.pyplot as plt
import os

def plot_model_metrics_before_after(
    before_metrics,
    after_metrics,
    graph_dir
):

    metrics_names=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    before_values=[
        before_metrics.get("accuracy",0),
        before_metrics.get("precision",0),
        before_metrics.get("recall",0),
        before_metrics.get("f1_score",0)
    ]

    after_values=[
        after_metrics.get("accuracy",0),
        after_metrics.get("precision",0),
        after_metrics.get("recall",0),
        after_metrics.get("f1_score",0)
    ]

    x=range(len(metrics_names))

    plt.figure(figsize=(8,5))

    plt.bar(
        [i-0.2 for i in x],
        before_values,
        width=0.4,
        label="Before"
    )

    plt.bar(
        [i+0.2 for i in x],
        after_values,
        width=0.4,
        label="After"
    )

    plt.xticks(x,metrics_names)

    plt.ylabel("Score")

    plt.title(
        "Model Metrics Before vs After"
    )

    plt.legend()

    output_path=os.path.join(
        graph_dir,
        "model_metrics_before_after.png"
    )

    plt.savefig(output_path)

    plt.close()

    print(
        f"\nGraph Saved : "
        f"{output_path}"
    )