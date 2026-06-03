from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)


def train_model(
    X_train,
    y_train,
    task_type
):

    # =====================================================
    # CLASSIFICATION MODEL
    # =====================================================

    if task_type == "classification":

        model = RandomForestClassifier(

            n_estimators=200,

            random_state=42
        )

    # =====================================================
    # REGRESSION MODEL
    # =====================================================

    else:

        model = RandomForestRegressor(

            n_estimators=200,

            random_state=42
        )

    # =====================================================
    # TRAIN MODEL
    # =====================================================

    model.fit(
        X_train,
        y_train
    )

    return model