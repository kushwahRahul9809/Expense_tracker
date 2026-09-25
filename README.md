# 💰 Smart Expense Tracker

A Flask web app for tracking personal expenses, with an AI model that automatically
categorizes each transaction from its description — and gets smarter the more you use it.

## Features

- Add expenses with description, amount, and category
- **AI auto-categorization**: leave category on "Auto" and a machine learning model
  (TF-IDF + Naive Bayes) predicts the category from the description text, with a
  live confidence preview as you type
- The model **retrains on your real data** after every new entry, so accuracy
  improves over time instead of staying static
- **Dashboard** with total spend, top category, a pie chart of spend by category,
  and a bar chart of monthly spending trend (built with pandas + matplotlib)
- Clean, responsive UI with color-coded category badges

## Tech Stack

| Layer | Tool |
|---|---|
| Backend framework | Flask |
| Database / ORM | SQLite + Flask-SQLAlchemy |
| Data analysis | pandas |
| Charts | matplotlib |
| AI / ML | scikit-learn (TfidfVectorizer + MultinomialNB) |
| Frontend | Jinja2 templates, vanilla CSS + JS |
| Production server | Gunicorn |

## Project Structure

```
expense_tracker/
├── app.py                 # Flask routes and app setup
├── models.py               # SQLAlchemy Transaction model
├── ml_categorizer.py       # AI model: training, saving, predicting categories
├── requirements.txt
├── Procfile                 # For deployment (gunicorn)
├── .gitignore
├── static/
│   └── style.css            # App styling
└── templates/
    ├── base.html             # Shared layout + navbar
    ├── index.html            # Transaction list
    ├── add.html               # Add-expense form with live AI suggestion
    └── dashboard.html         # Charts and spending summary
```

## Setup & Run Locally

1. **Clone / unzip the project**, then move into it:
   ```bash
   cd expense_tracker
   ```

2. **Install dependencies** (Python 3.9+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the initial AI model:**
   ```bash
   python ml_categorizer.py
   ```
   This creates `categorizer.joblib`, trained on seed example data.

4. **Run the app:**
   ```bash
   python app.py
   ```

5. Open **http://127.0.0.1:5000** in your browser.

## How the AI Categorization Works

- `ml_categorizer.py` builds a scikit-learn `Pipeline`:
  `TfidfVectorizer` (turns a description into a numeric feature vector, using
  1- and 2-word combinations) → `MultinomialNB` (a Naive Bayes classifier).
- It starts trained on a small seed dataset covering common categories:
  Food, Transport, Shopping, Bills, Entertainment, Other.
- Every time you add a real transaction, `app.py` calls `retrain_from_db()`,
  which re-trains the model on the seed data **plus every transaction you've
  logged** — so the model adapts to your actual spending habits and vocabulary.
- The `/predict` endpoint powers the live "AI suggests: ..." hint on the
  Add Expense page, called via a debounced `fetch()` request as you type.

## Deploying to Production

The dev server (`python app.py`) is for local use only. For a real deployment:

```bash
gunicorn app:app
```

The included `Procfile` (`web: gunicorn app:app`) is ready for platforms like
**Render**, **Railway**, or **Heroku**:

1. Push this project to a GitHub repo
2. Create a new Web Service on your platform, connect the repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Set environment variable `FLASK_DEBUG=0`

`app.py` reads `PORT` and `FLASK_DEBUG` from environment variables automatically,
so no code changes are needed between local and production use.

## Notes & Ideas for Extending

- Swap SQLite for PostgreSQL for a multi-user production setup
- Add user accounts/login so multiple people can track separately
- Replace Naive Bayes with an LLM-based categorizer (e.g. via the Claude API)
  for more nuanced, context-aware categorization
- Add budget limits per category with alerts
- Export transactions to CSV/Excel

## License

Free to use and modify for personal or educational purposes.
