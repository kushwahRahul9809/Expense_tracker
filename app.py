import base64
import io
import os

import matplotlib
matplotlib.use("Agg")  # no GUI backend needed on a server
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

from models import db, Transaction
from ml_categorizer import predict_with_confidence, train_and_save, CATEGORIES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(BASE_DIR, 'expenses.db')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

with app.app_context():
    db.create_all()


@app.route("/")
def index():
    transactions = Transaction.query.order_by(Transaction.date.desc()).all()
    return render_template("index.html", transactions=transactions, active="index")


@app.route("/add", methods=["GET", "POST"])
def add():
    suggestion, confidence = None, None

    if request.method == "POST":
        description = request.form["description"].strip()
        amount = float(request.form["amount"])
        category = request.form.get("category")

        # If the user didn't pick a category, let the AI model choose one
        if not category or category == "auto":
            category, _ = predict_with_confidence(description)

        txn = Transaction(description=description, amount=amount, category=category)
        db.session.add(txn)
        db.session.commit()

        # Retrain the model on the growing real dataset -> gets smarter over time
        retrain_from_db()

        return redirect(url_for("index"))

    # Live AI preview while typing (used by the /predict endpoint from JS)
    return render_template("add.html", categories=CATEGORIES, active="add")


@app.route("/predict", methods=["POST"])
def predict():
    """AJAX endpoint: given a description, return the AI's suggested category."""
    description = request.json.get("description", "")
    if not description.strip():
        return {"category": None, "confidence": 0}
    category, confidence = predict_with_confidence(description)
    return {"category": category, "confidence": confidence}


def retrain_from_db():
    transactions = Transaction.query.all()
    extra_data = [(t.description, t.category) for t in transactions]
    train_and_save(extra_data)


@app.route("/dashboard")
def dashboard():
    transactions = Transaction.query.all()

    if not transactions:
        return render_template("dashboard.html", has_data=False, active="dashboard")

    df = pd.DataFrame([t.to_dict() for t in transactions])
    df["date"] = pd.to_datetime(df["date"])

    total_spent = round(df["amount"].sum(), 2)
    by_category = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    by_month = df.groupby("month")["amount"].sum().sort_index()

    pie_chart = make_pie_chart(by_category)
    bar_chart = make_bar_chart(by_month)

    top_category = by_category.idxmax()

    return render_template(
        "dashboard.html",
        has_data=True,
        total_spent=total_spent,
        top_category=top_category,
        pie_chart=pie_chart,
        bar_chart=bar_chart,
        category_table=by_category.round(2).to_dict(),
        active="dashboard",
    )


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_pie_chart(by_category):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(by_category.values, labels=by_category.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Spending by Category")
    return fig_to_base64(fig)


def make_bar_chart(by_month):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(by_month.index, by_month.values, color="#4C72B0")
    ax.set_title("Monthly Spending Trend")
    ax.set_ylabel("Amount")
    plt.xticks(rotation=45)
    return fig_to_base64(fig)


if __name__ == "__main__":
    # Local development only. In production, gunicorn imports `app` directly
    # (see Procfile: "gunicorn app:app") and this block never runs.
    debug_mode = os.environ.get("FLASK_DEBUG", "1") == "1"
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=debug_mode, port=port)
