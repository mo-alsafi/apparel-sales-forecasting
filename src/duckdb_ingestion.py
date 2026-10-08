import os
import duckdb


DB_PATH = "data/processed/apparel_forecasting.duckdb"
RAW_DIR = "data/raw"


def initialize_database():
    os.makedirs("data/processed", exist_ok=True)
    conn = duckdb.connect(DB_PATH)
    
    conn.execute("""
        CREATE OR REPLACE TABLE raw_apparel_sales_nsa AS
        SELECT 
            CAST(DATE AS DATE) AS observation_date,
            CAST(MRTSSM448USN AS DOUBLE) AS apparel_sales_nsa
        FROM read_csv_auto('data/raw/MRTSSM448USN.csv', NULLSTR=['.', '']);
        
        CREATE OR REPLACE TABLE raw_apparel_sales_sa AS
        SELECT
            CAST(DATE AS DATE) AS observation_date,
            CAST(MRTSSM448USS AS DOUBLE) AS apparel_sales_sa
        FROM read_csv_auto('data/raw/MRTSSM448USS.csv', NULLSTR=['.', '']);
        
        CREATE OR REPLACE TABLE raw_cpi AS
        SELECT
            CAST(DATE AS DATE) AS observation_date,
            CAST(CPIAUCSL AS DOUBLE) AS cpi
        FROM read_csv_auto('data/raw/CPIAUCSL.csv', NULLSTR=['.', '']);
        
        CREATE OR REPLACE TABLE raw_unemployment_rate AS
        SELECT
            CAST(DATE AS DATE) AS observation_date,
            CAST(UNRATE AS DOUBLE) AS raw_unemployment_rate
        FROM read_csv_auto('data/raw/UNRATE.csv', NULLSTR=['.', '']);
    """)


    conn.execute("""
        CREATE OR REPLACE TABLE monthly_macro_apparel AS
        SELECT
            nsa.observation_date AS observation_date,
            nsa.apparel_sales_nsa AS apparel_sales_nsa,
            sa.apparel_sales_sa AS apparel_sales_sa,
            c.cpi AS cpi,
            u.raw_unemployment_rate AS un_rate
        FROM raw_apparel_sales_nsa nsa
        LEFT JOIN raw_apparel_sales_sa sa ON nsa.observation_date = sa.observation_date
        LEFT JOIN raw_cpi c ON nsa.observation_date = c.observation_date
        LEFT JOIN raw_unemployment_rate u ON nsa.observation_date = u.observation_date
        ORDER BY nsa.observation_date ASC;
    """)
    print("DuckDB processing finished. Unified created: monthly_marco_apparel")
    
    
    yoy_sales_growth = conn.sql("""
        WITH calculated_growth AS (
            SELECT
                observation_date,
                apparel_sales_nsa,
                LAG(apparel_sales_nsa, 12) OVER (ORDER BY observation_date ASC) AS sales_prior_year,
                ROUND(
                    (apparel_sales_nsa - LAG(apparel_sales_nsa, 12) OVER (ORDER BY observation_date ASC))
                    / LAG(apparel_sales_nsa, 12) OVER (ORDER BY observation_date ASC) * 100, 2
                ) AS yoy_growth_percentage    
            FROM monthly_macro_apparel
        )
        SELECT * 
        FROM calculated_growth
        WHERE observation_date >= '1993-01-01'
        ORDER BY observation_date ASC
    """).df()
    print("===YOY SALES GROWTH===\n", yoy_sales_growth.head(10))
    
    
    yearly_moving_avg_sales = conn.execute("""
        WITH yearly_sales_moving_avg AS (
            SELECT 
                observation_date,
                apparel_sales_nsa,
                ROUND(
                    AVG(apparel_sales_nsa) OVER (
                        ORDER BY observation_date ASC
                        ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
                    ), 2
                ) as moving_yearly_avg_sales
                FROM monthly_macro_apparel
        )
        SELECT 
            *
            FROM yearly_sales_moving_avg
            WHERE observation_date >= '1993-01-01';
    """).df()
    print("===YEARLY MOVING AVG SALES===\n", yearly_moving_avg_sales.head(20))
    
    
    rank_monthly_sales = conn.execute("""
        WITH RankMonthlySales AS (
            SELECT 
                EXTRACT(YEAR FROM observation_date) sales_year,
                EXTRACT(MONTH FROM observation_date) sales_month,
                observation_date,
                apparel_sales_nsa,
                ROW_NUMBER() OVER (
                    PARTITION BY EXTRACT(YEAR FROM observation_date)
                    ORDER BY apparel_sales_nsa DESC
                ) AS rank_highest,
                ROW_NUMBER() OVER (
                    PARTITION BY EXTRACT(YEAR FROM observation_date)
                    ORDER BY apparel_sales_nsa
                ) AS rank_lowest
            FROM monthly_macro_apparel
        )
        
        SELECT 
            sales_year,
            MAX(CASE WHEN rank_highest = 1 THEN sales_month END) AS peak_month,
            MAX(CASE WHEN rank_highest = 1 THEN apparel_sales_nsa END) AS peak_sales,
            MAX(CASE WHEN rank_lowest = 1 THEN sales_month END) AS trough_month,
            MAX(CASE WHEN rank_lowest = 1 THEN apparel_sales_nsa END) AS trough_sales
        FROM RankMonthlySales
        GROUP BY sales_year
        ORDER BY sales_year DESC;
    """).df()
    print("===RANKED MONTHLY SALES==\n", rank_monthly_sales.head(20))
    
    
    conn.close()
    
    
if __name__ == "__main__":
    initialize_database()