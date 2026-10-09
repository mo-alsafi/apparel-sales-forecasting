
WITH yearly_moving_avg AS(
	SELECT
		observation_date,
        apparel_sales_nsa AS sales,
        ROUND(
			AVG(apparel_sales_nsa) OVER (
				ORDER BY observation_date ASC
				ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
			), 2
		) AS moving_avg_yearly_sales
	FROM monthly_macro_apparel
)
SELECT
	* 
FROM yearly_moving_avg
WHERE observation_date >= '1993-01-01';