WITH yoy_sales_growth AS (
	SELECT 
		`DATE` AS observation_date,
        MRTSSM448USN AS sales,
		LAG(MRTSSM448USN, 12) OVER (ORDER BY `DATE` ASC) as sales_prior_year,
        ROUND(
			(MRTSSM448USN - LAG(MRTSSM448USN, 12) OVER (ORDER BY `DATE` ASC))
            / LAG(MRTSSM448USN, 12) OVER (ORDER BY `DATE` ASC) * 100, 2
        ) AS yoy_growth_percentage
	FROM raw_apparel_sales_nsa
)
SELECT 
	*
FROM yoy_sales_growth
WHERE observation_date >= '1993-01-01'
ORDER BY observation_date ASC;