/* 1. overall churn rate */
select
count(*) as total_customers,
sum(case when Churn = 'Yes' then 1 else 0 end)
as churned_customers,
round(
100.0 * sum(case when Churn = 'Yes' then 1 else 0 end)
/count(*),
2
)as churn_percentage
from customers;

/* 2.churn rate by contract */
select
Contract,
count(*) as total_customers,
sum(case when Churn ='Yes' then 1 else 0 end)
as churned_customers,
round(
100.0 *sum(case when Churn ='Yes' then 1 else 0 end)
/count(*),
2
)as churn_percentage
from customers
group by Contract
order by churn_percentage desc;

/* 3.monthly revenue by churn status */
select
Churn,
count(*) as customer_count,
round(sum(MonthlyCharges),2) as monthly_charges_total,
round(avg(MonthlyCharges),2) as average_monthly_charges
from customers
group by Churn;

/* 4.rank customers by tenure within each contract */
with ranked_customers as(
select
customerID,
Contract,
tenure,
Churn,
row_number() over(
partition by Contract
order by tenure desc
) as tenure_rank
from customers
) 
select *
from ranked_customers
where tenure_rank <= 10;