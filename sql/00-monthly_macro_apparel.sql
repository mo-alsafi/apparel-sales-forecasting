CREATE OR REPLACE TABLE monthly_macro_apparel AS
WITH date_bounds AS (
    SELECT
        LEAST(
            (SELECT MIN("DATE") FROM raw_apparel_sales_nsa),
            (SELECT MIN("DATE") FROM raw_cpi)
        ) AS min_date,
        GREATEST(
            (SELECT MAX("DATE") FROM raw_apparel_sales_nsa),
            (SELECT MAX("DATE") FROM raw_cpi),
            (SELECT MAX("DATE") FROM raw_unemployment_rate)
        ) AS max_date
),
monthly_spine AS (
    SELECT
        CAST(unnest(generate_series(min_date, max_date, INTERVAL '1 month')) AS DATE) AS observation_date
    FROM date_bounds
),
raw_joined AS (
    SELECT
        ms.observation_date,
        nsa.apparel_sales_nsa,
        sa.apparel_sales_sa,
        c.CPIAUCSL AS cpi_raw,
        u.UNRATE AS unrate_raw
    FROM monthly_spine ms
    LEFT JOIN raw_apparel_sales_nsa nsa ON ms.observation_date = nsa."DATE"
    LEFT JOIN raw_apparel_sales_sa sa ON ms.observation_date = sa."DATE"
    LEFT JOIN raw_cpi c ON ms.observation_date = c."DATE"
    LEFT JOIN raw_unemployment_rate u ON ms.observation_date = u."DATE"
)
SELECT
    observation_date,
    apparel_sales_nsa,
    apparel_sales_sa,
    CASE
        WHEN cpi_raw IS NULL AND observation_date < (SELECT MAX("DATE") FROM raw_cpi) THEN
            (LAG(cpi_raw, 1) OVER (ORDER BY observation_date) + LEAD(cpi_raw, 1) OVER (ORDER BY observation_date)) / 2.0
        ELSE cpi_raw
    END AS cpi,
    CASE
        WHEN cpi_raw IS NULL AND observation_date < (SELECT MAX("DATE") FROM raw_cpi) THEN 1
        ELSE 0
    END AS cpi_is_imputed,
    CASE
        WHEN unrate_raw IS NULL AND observation_date < (SELECT MAX("DATE") FROM raw_unemployment_rate) THEN
            (LAG(unrate_raw, 1) OVER (ORDER BY observation_date) + LEAD(unrate_raw, 1) OVER (ORDER BY observation_date)) / 2.0
        ELSE unrate_raw
    END AS unrate,
    CASE
        WHEN unrate_raw IS NULL AND observation_date < (SELECT MAX("DATE") FROM raw_unemployment_rate) THEN 1
        ELSE 0
    END AS unrate_is_imputed
FROM raw_joined
ORDER BY observation_date ASC;