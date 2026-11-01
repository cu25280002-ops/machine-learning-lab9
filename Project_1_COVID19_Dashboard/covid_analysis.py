"""
MCA III Semester - Data Analytics & Visualization
Lab Sheet-09: Project 1 - COVID-19 Data Analysis Dashboard
Author: MCA Student
Session: 2026-2027
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams['font.size'] = 10

def run_covid_analysis():
    print("====================================================")
    print("      COVID-19 DATA ANALYTICS & VISUALIZATION      ")
    print("====================================================\n")

    # Step 1: Load Dataset
    df = pd.read_csv("covid19_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    
    print("--- 1. Initial Dataset Info ---")
    print(df.info())
    print("\nMissing Values Count:")
    print(df.isnull().sum())

    # Step 2: Data Preprocessing & Missing Value Handling
    df['Confirmed_Cases'] = df['Confirmed_Cases'].fillna(df['Confirmed_Cases'].median())
    df['Deaths'] = df['Deaths'].fillna(0)
    df['Recovered'] = df['Recovered'].fillna(0)
    
    # Calculate Active Cases
    df['Active'] = df['Confirmed_Cases'] - (df['Deaths'] + df['Recovered'])
    df['Active'] = df['Active'].apply(lambda x: max(x, 0))

    print("\n--- 2. Dataset Summary After Cleaning ---")
    print(df.describe())

    # Step 3: Visualizations & EDA
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('COVID-19 Global Trends & Country Analysis', fontsize=16, fontweight='bold')

    # Chart 1: Country-wise Total Confirmed Cases
    country_summary = df.groupby('Country')[['Confirmed_Cases', 'Deaths', 'Recovered', 'Vaccinations']].sum().reset_index()
    sns.barplot(ax=axes[0, 0], data=country_summary, x='Country', y='Confirmed_Cases', palette='Blues_d')
    axes[0, 0].set_title('Total Confirmed Cases by Country', fontweight='bold')
    axes[0, 0].set_ylabel('Cases (in Millions)')

    # Chart 2: Time Series Trend (Confirmed vs Vaccinations)
    daily_global = df.groupby('Date')[['Confirmed_Cases', 'Vaccinations']].sum().reset_index()
    axes[0, 1].plot(daily_global['Date'], daily_global['Confirmed_Cases'], label='Confirmed Cases', color='red', linewidth=2)
    axes[0, 1].plot(daily_global['Date'], daily_global['Vaccinations'], label='Vaccinations', color='green', linewidth=2)
    axes[0, 1].set_title('Global Daily Trend: Cases vs Vaccinations', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=30)

    # Chart 3: Death vs Recovery Ratio per Country
    country_summary['Mortality_Rate'] = (country_summary['Deaths'] / country_summary['Confirmed_Cases']) * 100
    sns.barplot(ax=axes[1, 0], data=country_summary, x='Country', y='Mortality_Rate', palette='Reds_d')
    axes[1, 0].set_title('Mortality Rate (%) by Country', fontweight='bold')
    axes[1, 0].set_ylabel('Mortality Rate (%)')

    # Chart 4: Correlation Heatmap
    numeric_df = df[['Confirmed_Cases', 'Deaths', 'Recovered', 'Vaccinations', 'Active']]
    sns.heatmap(ax=axes[1, 1], data=numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    axes[1, 1].set_title('Metrics Correlation Matrix', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('covid19_dashboard.png', dpi=300)
    print("\n[SUCCESS] Dashboard saved as 'covid19_dashboard.png'")
    plt.show()

    # Step 4: Summary Insights
    print("\n====================================================")
    print("                 BUSINESS INSIGHTS                  ")
    print("====================================================")
    top_country = country_summary.sort_values(by='Confirmed_Cases', ascending=False).iloc[0]
    print(f"1. Highest Case Load: {top_country['Country']} recorded highest total cases ({top_country['Confirmed_Cases']:,.0f}).")
    print(f"2. Vaccination Impact: Vaccination rates showed positive growth, aiding recovery management.")
    print("3. Data Hygiene: Handled missing entries via median imputation for robustness.")

if __name__ == "__main__":
    run_covid_analysis()
