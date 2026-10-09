WITH rolling_yearly_avg_sales AS(
	SELECT
		observation_date,
        apparel_sales_nsa,
        ROUND(
			AVG(apparel_sales_nsa) OVER (
				ORDER BY observation_date ASC
				ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
			), 2
		) AS rolling_yearly_avg_sales
	FROM monthly_macro_apparel
)
SELECT
	observation_date,
    apparel_sales_nsa,
    rolling_yearly_avg_sales
FROM rolling_yearly_avg_sales
WHERE observation_date >= '1993-01-01'
	AND apparel_sales_nsa IS NOT NULL
ORDER BY observation_date ASC;
