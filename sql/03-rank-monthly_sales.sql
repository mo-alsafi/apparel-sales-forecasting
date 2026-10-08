WITH RankMonthlySales AS (
	SELECT
		EXTRACT(YEAR FROM `DATE`) AS sales_year,
		EXTRACT(MONTH FROM `DATE`) AS sales_month,
        `DATE` AS observation_date,
        MRTSSM448USN AS sales,
        ROW_NUMBER() OVER(
			PARTITION BY EXTRACT(YEAR FROM `DATE`)
            ORDER BY MRTSSM448USN DESC
        ) AS rank_highest,
        ROW_NUMBER() OVER(
			PARTITION BY EXTRACT(YEAR FROM `DATE`)
            ORDER BY MRTSSM448USN ASC
        ) AS rank_lowest
	FROM raw_apparel_sales_nsa
)
SELECT 
	sales_year,
    MAX(CASE WHEN rank_highest = 1 THEN sales_month END) AS highest_month, -- case will chech every month in the year (cause group by sales_year) if = 1 return sales if not return null so we use max to get rid of nulls
    MAX(CASE WHEN rank_highest = 1 THEN sales END) AS highest_sales,
    MAX(CASE WHEN rank_lowest = 1 THEN sales_month END) AS lowest_month,
    MAX(CASE WHEN rank_lowest = 1 THEN sales END) AS lowest_sales
FROM RankMonthlySales
GROUP BY sales_year
ORDER BY sales_year DESC;