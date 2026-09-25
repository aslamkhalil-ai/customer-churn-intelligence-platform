from flask import Flask, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)


# =====================================================
# LOAD MODEL FILES
# =====================================================

model = joblib.load("churn_model (1).joblib")

scaler = joblib.load("scaler (2).pkl")

feature_names = joblib.load("feature_names.pkl")


# =====================================================
# LOAD CUSTOMER DATA
# =====================================================

df = pd.read_csv(
    "Bank Customer Churn Prediction.csv"
)

age_churn = (
    df.assign(
        age_group=pd.cut(
            df["age"],
            bins=[0, 25, 35, 45, 55, 100],
            labels=[
                "18-25",
                "26-35",
                "36-45",
                "46-55",
                "56+"
            ]
        )
    )
    .groupby("age_group", observed=False)["churn"]
    .sum()
    .to_dict()
)


products_churn = (
    df.groupby("products_number")["churn"]
    .sum()
    .to_dict()
)

# =====================================================
# HOME ROUTE
# =====================================================

@app.route("/", methods=["GET", "POST"])
def home():

    # -------------------------------------------------
    # DEFAULT VALUES
    # -------------------------------------------------

    prediction = None

    probability = None

    risk_level = None


    # =================================================
    # DASHBOARD STATISTICS
    # =================================================

    total_customers = len(df)

    churned_customers = int(
        df["churn"].sum()
    )

    stayed_customers = (
        total_customers -
        churned_customers
    )

    if total_customers > 0:

        churn_rate = round(
            (
                churned_customers /
                total_customers
            ) * 100,
            2
        )

    else:

        churn_rate = 0


    # =================================================
    # ANALYTICS DATA
    # =================================================

    country_churn = (
        df.groupby("country")["churn"]
        .sum()
        .to_dict()
    )


    gender_churn = (
        df.groupby("gender")["churn"]
        .sum()
        .to_dict()
    )


    active_churn = (
        df.groupby("active_member")["churn"]
        .sum()
        .to_dict()
    )


    # =================================================
    # CUSTOMER FILTERS
    # =================================================

    search = request.args.get(
        "search",
        ""
    ).strip()


    country = request.args.get(
        "country",
        ""
    ).strip()


    status = request.args.get(
        "status",
        ""
    ).strip()


    # Start with complete dataset

    filtered_df = df.copy()


    # -------------------------------------------------
    # SEARCH CUSTOMER ID
    # -------------------------------------------------

    if search != "":

        filtered_df = filtered_df[
            filtered_df["customer_id"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]


    # -------------------------------------------------
    # COUNTRY FILTER
    # -------------------------------------------------

    if country != "":

        filtered_df = filtered_df[
            filtered_df["country"]
            .astype(str)
            == country
        ]


    # -------------------------------------------------
    # CHURN STATUS FILTER
    # -------------------------------------------------

    if status != "":

        filtered_df = filtered_df[
            filtered_df["churn"]
            .astype(str)
            == status
        ]


    # =================================================
    # PAGINATION
    # =================================================

    per_page = 10


    total_filtered = len(
        filtered_df
    )


    page = request.args.get(
        "page",
        default=1,
        type=int
    )


    if page < 1:

        page = 1


    total_pages = max(
        1,
        (
            total_filtered +
            per_page -
            1
        ) // per_page
    )


    if page > total_pages:

        page = total_pages


    start = (
        page - 1
    ) * per_page


    end = (
        start +
        per_page
    )


    customers = (
        filtered_df
        .iloc[start:end]
        .to_dict(
            orient="records"
        )
    )


    # =================================================
    # CUSTOMER CHURN PREDICTION
    # =================================================

    if request.method == "POST":

        # ---------------------------------------------
        # GET FORM DATA
        # ---------------------------------------------

        credit_score = float(
            request.form["credit_score"]
        )


        prediction_country = request.form[
            "country"
        ]


        gender = request.form[
            "gender"
        ]


        age = float(
            request.form["age"]
        )


        tenure = float(
            request.form["tenure"]
        )


        balance = float(
            request.form["balance"]
        )


        products_number = float(
            request.form["products_number"]
        )


        credit_card = float(
            request.form["credit_card"]
        )


        active_member = float(
            request.form["active_member"]
        )


        estimated_salary = float(
            request.form["estimated_salary"]
        )


        # ---------------------------------------------
        # CREATE DATAFRAME
        # ---------------------------------------------

        data = pd.DataFrame([{

            "credit_score":
                credit_score,

            "country":
                prediction_country,

            "gender":
                gender,

            "age":
                age,

            "tenure":
                tenure,

            "balance":
                balance,

            "products_number":
                products_number,

            "credit_card":
                credit_card,

            "active_member":
                active_member,

            "estimated_salary":
                estimated_salary

        }])


        # ---------------------------------------------
        # ENCODE CATEGORICAL FEATURES
        # ---------------------------------------------

        data = pd.get_dummies(

            data,

            columns=[
                "country",
                "gender"
            ],

            drop_first=True

        )


        # ---------------------------------------------
        # MATCH TRAINING FEATURES
        # ---------------------------------------------

        data = data.reindex(

            columns=feature_names,

            fill_value=0

        )


        # ---------------------------------------------
        # SCALE DATA
        # ---------------------------------------------

        data_scaled = scaler.transform(
            data
        )


        # ---------------------------------------------
        # MODEL PREDICTION
        # ---------------------------------------------

        prediction = model.predict(
            data_scaled
        )[0]


        # ---------------------------------------------
        # CHURN PROBABILITY
        # ---------------------------------------------

        if hasattr(
            model,
            "predict_proba"
        ):

            probability = round(

                model.predict_proba(
                    data_scaled
                )[0][1] * 100,

                1

            )


            # -----------------------------------------
            # RISK LEVEL
            # -----------------------------------------

            if probability >= 70:

                risk_level = "HIGH RISK"


            elif probability >= 40:

                risk_level = "MEDIUM RISK"


            else:

                risk_level = "LOW RISK"


    # =================================================
    # SEND DATA TO HTML
    # =================================================

    return render_template(

        "index.html",

        # Prediction
        prediction=prediction,

        probability=probability,

        risk_level=risk_level,


        # Dashboard
        total_customers=total_customers,

        churned_customers=churned_customers,

        stayed_customers=stayed_customers,

        churn_rate=churn_rate,


        # Analytics
        country_churn=country_churn,

        gender_churn=gender_churn,

        active_churn=active_churn,
        age_churn=age_churn,
        products_churn=products_churn,


        # Customers
        customers=customers,

        page=page,

        total_pages=total_pages,

        total_filtered=total_filtered,

        per_page=per_page,


        # Filters
        search=search,

        country=country,

        status=status

    )


# =====================================================
# RUN FLASK
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )