/*+ ETLM {
depend:{
add:[
{name:"VAP_AGG.ENDAL_PVC_SEGMENTS_TRAFFIC"},
{name:"VAP_AGG.ENDAL_PVC_SEGMENTS_CONV"},
{name:"VAP_AGG.ENDAL_PVC_ELIGIBLE_CAMPAIGNS"}
]
}
}*/
with ads_data as (
select 
st.base_seg_id,
st.dim_date,
st.marketplace_id,
st.ad_cfid,
st.campaign_cfid,
st.start_dt_utc,
st.end_dt_utc,
st.dim_advertiser_id,
st.advertiser_name,
st.campaign_name,
st.entity_id,
st.seg_name,
st.targ_method,
sum(impressions) as impressions,
sum(clicks) as clicks,
case when sum(nvl(subscription_free_trial_click_purchases,0))+ sum(nvl(subscription_free_trial_view_purchases,0))
    +SUM(nvl(subscription_win_back_view_purchases, 0)
      + nvl(subscription_initial_promotion_view_purchases, 0)
      + nvl(subscription_win_back_click_purchases, 0)
      + nvl(subscription_initial_promotion_click_purchases, 0))>0 then sum(nvl(subscription_free_trial_click_purchases,0))+ sum(nvl(subscription_free_trial_view_purchases,0))
    +SUM(nvl(subscription_win_back_view_purchases, 0)
      + nvl(subscription_initial_promotion_view_purchases, 0)
      + nvl(subscription_win_back_click_purchases, 0)
      + nvl(subscription_initial_promotion_click_purchases, 0))

 when sum(nvl(amazon_pay_initial_click_purchases,0))+ sum(nvl(amazon_pay_initial_view_purchases,0)) > 0 then sum(nvl(amazon_pay_initial_click_purchases,0))+ sum(nvl(amazon_pay_initial_view_purchases,0)) end
 as conversions

 from
VAP_AGG.ENDAL_PVC_SEGMENTS_TRAFFIC st
LEFT JOIN VAP_AGG.ENDAL_PVC_SEGMENTS_CONV sc
ON st.base_seg_id = sc.base_seg_id
and st.dim_date= sc.dim_Date
and st.marketplace_id=sc.marketplace_id
and st.ad_cfid = sc.ad_cfid
and st.targ_method = sc.targ_method
and st.seg_name = sc.seg_name
where st.targ_method = 'untargeted'  -- Filter on only untargeted segments
group by 
st.base_seg_id,
st.dim_date,
st.marketplace_id,
st.ad_cfid,
st.campaign_cfid,
st.start_dt_utc,
st.end_dt_utc,
st.dim_advertiser_id,
st.advertiser_name,
st.campaign_name,
st.entity_id,
st.seg_name,
st.targ_method
),
metric_ads_conversions as (
select a.*,
avg(cost_per_conversion) over (partition by ad_cfid rows between unbounded preceding and unbounded following) as avg_cpa,
stddev(cost_per_conversion) over (partition by ad_cfid rows between unbounded preceding and unbounded following) as stddev_cpa
from (
select 
st.base_seg_id,
st.marketplace_id,
st.ad_cfid,
st.campaign_cfid,
st.start_dt_utc,
st.end_dt_utc,
st.dim_advertiser_id,
st.advertiser_name,
st.campaign_name,
st.entity_id,
st.seg_name,
st.targ_method,
property_type,
targeting_combined,
sum(st.impressions) as impressions,
sum(clicks) as clicks,
sum(conversions) as conversions,
case when sum(conversions) > 0 then nvl(((sum(st.impressions)/1000)*(max(a.revenue)/(max(a.total_impressions)/1000)))/sum(conversions),0) else 0 end as cost_per_conversion,
nvl((sum(st.clicks)*1000000)/sum(impressions),0) as ctr
from 
ads_data st
JOIN VAP_AGG.ENDAL_PVC_ELIGIBLE_CAMPAIGNS a ON st.ad_cfid = a.cfid
    where st.impressions > 100
    and conversions > 0
group by  
st.base_seg_id,
st.marketplace_id,
st.ad_cfid,
st.campaign_cfid,
st.start_dt_utc,
st.end_dt_utc,
st.dim_advertiser_id,
st.advertiser_name,
st.campaign_name,
st.entity_id,
st.seg_name,
st.targ_method,
property_type,
targeting_combined
) a
),
metric_ads_clicks as (
select a.*,
avg(ctr) over (partition by ad_cfid rows between unbounded preceding and unbounded following) as avg_cpa,
stddev(ctr) over (partition by ad_cfid rows between unbounded preceding and unbounded following) as stddev_cpa
from (
select 
st.base_seg_id,
st.marketplace_id,
st.ad_cfid,
st.campaign_cfid,
st.start_dt_utc,
st.end_dt_utc,
st.dim_advertiser_id,
st.advertiser_name,
st.campaign_name,
st.entity_id,
st.seg_name,
st.targ_method,
property_type,
targeting_combined,
sum(st.impressions) as impressions,
sum(clicks) as clicks,
sum(conversions) as conversions,
case when sum(conversions) > 0 then nvl(((sum(st.impressions)/1000)*(max(a.revenue)/(max(a.total_impressions)/1000)))/sum(conversions),0) else 0 end as cost_per_conversion,
nvl((sum(st.clicks)*1000000)/sum(impressions),0) as ctr
from 
ads_data st
JOIN VAP_AGG.ENDAL_PVC_ELIGIBLE_CAMPAIGNS a ON st.ad_cfid = a.cfid
    where st.impressions > 100
    and clicks > 0
group by  
st.base_seg_id,
st.marketplace_id,
st.ad_cfid,
st.campaign_cfid,
st.start_dt_utc,
st.end_dt_utc,
st.dim_advertiser_id,
st.advertiser_name,
st.campaign_name,
st.entity_id,
st.seg_name,
st.targ_method,
property_type,
targeting_combined
) a
)
,filter_metrics_data as ( --- Only consider segments 2 stddev away from mean
select a.*, last_value(type) over (partition by ad_cfid rows between unbounded preceding and unbounded following) as first_type
from (
select cc.*, 'Conv' as type, case when cost_per_conversion >= avg_cpa+(2*stddev_cpa) then 1 else 0 end as tbr from metric_ads_conversions cc
where avg_cpa > 0
--UNION ALL
--select c.*, 'Clicks' as typec,case when ctr < avg_cpa-(stddev_cpa) then 1 else 0 end as tbr from metric_ads_clicks c
--where avg_cpa > 0
) a
)

select entity_id as entityid
, null as customerid
, campaign_cfid as ordercfid
, ad_cfid as lineitemcfid
, 'SEGMENT_REMOVAL' as optimizationlever
, 'REMOVE_UNTARGETED_SEGMENT' as recommendedaction
, '{"TARGETING_STRING":"'|| targeting_Combined || '"}' as curdata
, '{"TARGETING_STRING":"'||listagg('s='||base_seg_id,' OR ') || '"}'  as newdata
,start_dt_utc as eligiblestartdate	
, end_dt_utc as eligibleenddate	
, 'PENDING' as status
, case when property_type = 'Class 1' then 'Class_I' else 'AAP' end as orderstrategy
, case when property_type = 'Class 1' then 'ORDER' else 'LINEITEM' end as updatelevel
, 'NA' as region
, 1 as marketplaceid
, case when property_type = 'Class 1' then 'FORECAST' else 'NO_FORECAST' end as updatestrategy
, 'PVC'as sourcetype
, sysdate as createdAt
, sysdate as updatedAt
from filter_metrics_data
where first_type = type 
and tbr = 1
and base_seg_id not in (32, 20, 21)
--and mod(datediff('day',to_date(start_dt_utc,'YYYYMMDD'),to_date('{RUN_DATE_YYYYMMDD}','YYYMMDD')),7)=1
group by entity_id, campaign_cfid, ad_cfid, targeting_combined, start_Dt_utc, end_dt_utc, property_type