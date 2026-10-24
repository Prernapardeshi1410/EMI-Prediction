import pandas as pd
import random

random.seed(42)

data = []

for _ in range(2000):
    income = random.randint(15000, 200000)
    loan_amount = random.randint(50000, 2000000)
    interest_rate = round(random.uniform(6.0, 18.0), 2)
    tenure_years = random.randint(1, 10)
    existing_emi = random.randint(0, 20000)

    months = tenure_years * 12
    monthly_rate = interest_rate / 1200

    if monthly_rate == 0:
        emi = loan_amount / months
    else:
        emi = (
            loan_amount * monthly_rate
            * (1 + monthly_rate) ** months
            / ((1 + monthly_rate) ** months - 1)
        )

    total_emi = emi + existing_emi
    emi_ratio = total_emi / income

    risk = "High" if emi_ratio > 0.50 else (
        "Medium" if emi_ratio > 0.30 else "Low"
    )

    data.append({
        "income": income,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "tenure_years": tenure_years,
        "existing_emi": existing_emi,
        "monthly_emi": round(emi, 2),
        "emi_to_income_ratio": round(emi_ratio, 3),
        "risk": risk
    })

df = pd.DataFrame(data)
df.to_csv("emi_dataset.csv", index=False)

print("Dataset created successfully!")
print("Total records:", len(df))
print(df.head())
