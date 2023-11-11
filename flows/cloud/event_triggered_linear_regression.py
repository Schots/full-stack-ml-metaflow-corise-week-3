from metaflow import FlowSpec, step, card, conda_base, current, Parameter, Flow, trigger
from metaflow.cards import Markdown, Table, Image, Artifact

URL = "https://outerbounds-datasets.s3.us-west-2.amazonaws.com/taxi/latest.parquet"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
FEATURES = ["tpep_pickup_datetime",
          "passenger_count",
          "RatecodeID",
          "trip_distance"]
TARGET = ["total_amount"]

@trigger(events=["s3"])
@conda_base(
    libraries={
        "pandas": "1.4.2",
        "pyarrow": "11.0.0",
        "scikit-learn": "1.1.2",
        "dirty_cat":"0.4.1"
    }
)
class TaxiFarePrediction(FlowSpec):
    data_url = Parameter("data_url", default=URL)

    @step
    def start(self):
        import pandas as pd
        
        self.data = pd.read_parquet(self.data_url)
        self.X = self.data[FEATURES]
        self.y = self.data[TARGET]
        
        self.next(self.vectorizer_assembly)
        
    @step
    def vectorizer_assembly(self):
        from dirty_cat import TableVectorizer
        from sklearn.preprocessing import OrdinalEncoder


        low_card_encoder = OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1)
        self.vectorizer = TableVectorizer(impute_missing="force",
                                          low_card_cat_transformer=low_card_encoder)

        self.next(self.regressor_model)

    @step
    def regressor_model(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.pipeline import make_pipeline

        self.model = make_pipeline(self.vectorizer,HistGradientBoostingRegressor())

        self.next(self.validate)

    def gather_sibling_flow_run_results(self):
        # storage to populate and feed to a Table in a Metaflow card
        rows = []

        # loop through runs of this flow
        for run in Flow(self.__class__.__name__):
            if run.id != current.run_id:
                if run.successful:
                    icon = "✅"
                    msg = "OK"
                    score = str(run.data.scores.mean())
                else:
                    icon = "❌"
                    msg = "Error"
                    score = "NA"
                    for step in run:
                        for task in step:
                            if not task.successful:
                                msg = task.stderr
                row = [
                    Markdown(icon),
                    Artifact(run.id),
                    Artifact(run.created_at.strftime(DATETIME_FORMAT)),
                    Artifact(score),
                    Markdown(msg),
                ]
                rows.append(row)
            else:
                rows.append(
                    [
                        Markdown("✅"),
                        Artifact(run.id),
                        Artifact(run.created_at.strftime(DATETIME_FORMAT)),
                        Artifact(str(self.scores.mean())),
                        Markdown("This run..."),
                    ]
                )
        return rows

    @card(type="corise")
    @step
    def validate(self):
        from sklearn.model_selection import cross_val_score

        self.scores = cross_val_score(self.model, self.X, self.y, cv=5)
        current.card.append(Markdown("# Taxi Fare Prediction Results"))
        current.card.append(
            Table(
                self.gather_sibling_flow_run_results(),
                headers=["Pass/fail", "Run ID", "Created At", "R^2 score", "Stderr"],
            )
        )
        self.next(self.end)

    @step
    def end(self):
        print(f"R2 Score:{self.scores.mean():0.2f}")
        print("Success!")


if __name__ == "__main__":
    TaxiFarePrediction()
