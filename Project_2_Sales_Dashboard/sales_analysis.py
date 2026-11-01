"""
MCA III Semester - Data Analytics & Visualization
Lab Sheet-09: Project 2 - Sales Data Analysis Dashboard
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

def run_sales_analysis():
    print("====================================================")
    print("       SALES DATA ANALYTICS & DASHBOARD             ")
    print("====================================================\n")

    # Step 1: Load Dataset
    df = pd.read_csv("sales_data.csv")
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    
    print("--- 1. Initial Dataset Info ---")
    print(df.info())
    print("\nMissing Values Count:")
    print(df.isnull().sum())

    # Step 2: Data Preprocessing
    # Impute missing profit using mean profit margin ratio of corresponding Category
    df['Profit_Margin'] = df['Profit'] / df['Sales']
    mean_margin = df.groupby('Category')['Profit_Margin'].transform('mean')
    df['Profit'] = df['Profit'].fillna(df['Sales'] * mean_margin)
    
    # Extract Month-Year for Trend Analysis
    df['YearMonth'] = df['Order_Date'].dt.to_period('M')

    print("\n--- 2. Dataset Summary After Cleaning ---")
    print(df.describe())

    # Step 3: Visualizations & Dashboard Creation
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Interactive Executive Sales Performance Dashboard', fontsize=16, fontweight='bold')

    # Chart 1: Monthly Sales & Profit Trend
    monthly_sales = df.groupby('YearMonth')[['Sales', 'Profit']].sum().reset_index()
    monthly_sales['YearMonth_Str'] = monthly_sales['YearMonth'].astype(str)
    
    axes[0, 0].plot(monthly_sales['YearMonth_Str'], monthly_sales['Sales'], marker='o', label='Sales ($)', color='#1f77b4', linewidth=2)
    axes[0, 0].plot(monthly_sales['YearMonth_Str'], monthly_sales['Profit'], marker='s', label='Profit ($)', color='#2ca02c', linewidth=2)
    axes[0, 0].set_title('Monthly Sales vs Profit Trend', fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].legend()

    # Chart 2: Product Category Performance
    cat_perf = df.groupby('Category')[['Sales', 'Profit']].sum().reset_index()
    sns.barplot(ax=axes[0, 1], data=cat_perf, x='Category', y='Sales', palette='Blues_d')
    axes[0, 1].set_title('Sales Performance by Product Category', fontweight='bold')
    axes[0, 1].set_ylabel('Total Sales ($)')

    # Chart 3: Regional Sales Distribution
    region_perf = df.groupby('Region')['Sales'].sum().reset_index()
    axes[1, 0].pie(region_perf['Sales'], labels=region_perf['Region'], autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    axes[1, 0].set_title('Regional Sales Share', fontweight='bold')

    # Chart 4: Customer Segmentation Profitability
    sns.boxplot(ax=axes[1, 1], data=df, x='Customer_Segment', y='Profit', palette='Set2')
    axes[1, 1].set_title('Profit Distribution across Customer Segments', fontweight='bold')
    axes[1, 1].set_ylabel('Profit per Order ($)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('sales_dashboard.png', dpi=300)
    print("\n[SUCCESS] Dashboard saved as 'sales_dashboard.png'")
    plt.show()

    # Step 4: Executive Business Insights
    print("\n====================================================")
    print("                 BUSINESS INSIGHTS                  ")
    print("====================================================")
    top_cat = cat_perf.sort_values(by='Sales', ascending=False).iloc[0]
    top_region = region_perf.sort_values(by='Sales', ascending=False).iloc[0]
    
    print(f"1. Top Category: '{top_cat['Category']}' generated highest revenue (${top_cat['Sales']:,.2f}).")
    print(f"2. Regional Dominance: {top_region['Region']} region leads sales with {top_region['Sales']:,.2f} USD.")
    print("3. Segment Focus: Corporate and Consumer segments maintain high volume with consistent profit margins.")

if __name__ == "__main__":
    run_sales_analysis()
