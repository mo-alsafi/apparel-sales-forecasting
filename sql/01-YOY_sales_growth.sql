-- YearOverYear Sales Growth
WITH yoy_sales AS (
	SELECT 
		observation_date,
        apparel_sales_nsa,
        LAG(apparel_sales_nsa, 12) OVER (ORDER BY observation_date ASC) AS sales_prior_year
	FROM monthly_macro_apparel
)
SELECT 
	observation_date,
    apparel_sales_nsa,
    sales_prior_year,
    ROUND(
		(apparel_sales_nsa - sales_prior_year) / NULLIF(sales_prior_year, 0) * 100, 2
    ) AS yoy_growth_percentage
FROM yoy_sales
WHERE observation_date >= '1993-01-01'
	AND apparel_sales_nsa IS NOT NULL
ORDER BY observation_date ASC;