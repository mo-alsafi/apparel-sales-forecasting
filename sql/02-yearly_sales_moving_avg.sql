WITH yearly_moving_avg AS(
	SELECT
		`DATE` as observation_date,
        MRTSSM448USN AS sales,
        ROUND(
			AVG(MRTSSM448USN) OVER (
				ORDER BY `DATE` ASC
				ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
			), 2
		) AS moving_avg_yearly_sales
	FROM raw_apparel_sales_nsa
)
SELECT
	* 
FROM yearly_moving_avg
WHERE observation_date >= '1993-01-01';