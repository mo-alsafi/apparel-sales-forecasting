
ALTER TABLE raw_apparel_sales_nsa MODIFY COLUMN `DATE` DATE;
ALTER TABLE raw_apparel_sales_nsa RENAME COLUMN MRTSSM448USN TO apparel_sales_nsa
ALTER TABLE raw_apparel_sales_nsa MODIFY COLUMN apparel_sales_nsa DOUBLE;

ALTER TABLE raw_apparel_sales_sa MODIFY COLUMN `DATE` DATE;
ALTER TABLE raw_apparel_sales_sa RENAME COLUMN MRTSSM448USS TO apparel_sales_sa
ALTER TABLE raw_apparel_sales_sa MODIFY COLUMN apparel_sales_sa DOUBLE;

ALTER TABLE raw_cpi MODIFY COLUMN `DATE` DATE;
ALTER TABLE raw_cpi MODIFY COLUMN CPIAUCSL DOUBLE;

ALTER TABLE raw_unemployment_rate MODIFY COLUMN `DATE` DATE;
ALTER TABLE raw_unemployment_rate MODIFY COLUMN UNRATE DOUBLE;


CREATE TABLE IF NOT EXISTS monthly_macro_apparel AS
WITH RECURSIVE date_bounds AS (
	SELECT
		LEAST(
			(SELECT MIN(`DATE`) FROM raw_apparel_sales_nsa),
            (SELECT MIN(`DATE`) FROM raw_cpi)
        ) AS min_date,
        GREATEST(
			(SELECT MAX(`DATE`) FROM raw_apparel_sales_nsa),
            (SELECT MAX(`DATE`) FROM raw_cpi),
            (SELECT MAX(`DATE`) FROM raw_unemployment_rate)
        ) AS max_date
),
monthly_spine AS (
	SELECT min_date FROM date_bounds
    UNION ALL 
    SELECT ms.min_date + INTERVAL 1 MONTH
    FROM monthly_spine ms
    CROSS JOIN date_bounds db
    WHERE ms.min_date < db.max_date
),
raw_joined AS (
	SELECT 
		ms.min_date AS observation_date,
        nsa.apparel_sales_nsa AS apparel_sales_nsa,
        sa.apparel_sales_sa AS apparel_sales_sa,
        c.CPIAUCSL AS cpi_raw,
        u.UNRATE AS unrate_raw
	FROM monthly_spine ms
    LEFT JOIN raw_apparel_sales_nsa nsa ON ms.min_date = nsa.`DATE`
	LEFT JOIN raw_apparel_sales_sa sa ON ms.min_date = sa.`DATE`
    LEFT JOIN raw_cpi c ON ms.min_date = c.`DATE`
    LEFT JOIN raw_unemployment_rate u ON ms.min_date = u.`DATE`
)
SELECT 
	observation_date,
    apparel_sales_nsa,
    apparel_sales_sa,
    
    -- CPI Interpolation
    CASE
		WHEN cpi_raw IS NULL AND observation_date < (SELECt MAX(`DATE`) FROM raw_cpi) THEN 
			(LAG(cpi_raw, 1) OVER (ORDER BY observation_date) + LEAD(cpi_raw, 1) OVER (ORDER BY observation_date)) / 2.0
		ELSE cpi_raw
	END AS cpi,
    CASE 
		WHEN cpi_raw IS NULL AND observation_date < (SELECT Max(`DAtE`) FROM raw_cpi) THEN 1
		ELSE 0
	END AS cpi_is_imputed,
    
    -- Unemployment Interpolation
    CASE 
		WHEN unrate_raw IS NULL AND observation_date < (SELECT MAX(`DATE`) FROM raw_unemployment_rate) THEN 
			(LAG(unrate_raw, 1) OVER (ORDER BY observation_date) + LEAD(unrate_raw, 1) OVER (ORDER BY observation_date)) / 2.0
		ELSE unrate_raw
	END AS unrate,
    CASE 
		WHEN unrate_raw IS NULL AND observation_date < (SELECT MAX(`DATE`) FROM raw_unemployment_rate) THEN 1
        ELSE 0
	END AS unrate_is_imputed
FROM raw_joined;