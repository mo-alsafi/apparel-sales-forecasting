-- Best and worst month per full calender year
WITH CompleteYears AS (
	SELECT EXTRACT(YEAR FROM observation_date) AS sales_year
	FROM monthly_macro_apparel
	WHERE apparel_sales_nsa IS NOT NULL
    GROUP BY sales_year
    HAVING COUNT(DISTINCT EXTRACT(MONTH FROM observation_date)) = 12
),
RankedMonthlySales AS (
	SELECT 
		EXTRACT(YEAR FROM observation_date) AS sales_year,
        EXTRACT(MONTH FROM observation_date) AS sales_month,
        observation_date,
        apparel_sales_nsa,
        ROW_NUMBER() OVER(
			PARTITION BY EXTRACT(YEAR FROM observation_date)
            ORDER BY apparel_sales_nsa DESC
        ) AS rank_highest,
        ROW_NUMBER() OVER (
			PARTITION BY EXTRACT(YEAR FROM observation_date)
            ORDER BY apparel_sales_nsa ASC
        ) AS rank_lowest
	FROM monthly_macro_apparel
    WHERE EXTRACT(YEAR FROM observation_date) IN (SELECT sales_year FROM CompleteYears)
)
SELECT
	sales_year,
    MAX(CASE WHEN rank_highest = 1 THEN sales_month END) AS peak_month,
    MAX(CASE WHEN rank_highest = 1 THEN apparel_sales_nsa END) AS peak_sales_millions,
    MAX(CASE WHEN rank_lowest = 1 THEN sales_month END) AS trough_month,
    MAX(CASE WHEN rank_lowest = 1 THEN apparel_sales_nsa END) AS trough_sales_millions
FROM RankedMonthlySales
GROUP BY sales_year
ORDER BY sales_year DESC;
