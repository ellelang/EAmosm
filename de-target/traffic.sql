/*+ ETLM {
    depend:{
       add:[
        {
          name:"adw_metrics_glue.WRT_DAILY_SEG_AD_TRAFFIC",
          age:{days:3}
      }              
    ]
}
} */
WITH filtered_ads as
--ad meet criteria 2,3,6,7
(
SELECT dads.dim_ad_id as ad_id
FROM 
	adw_metrics_glue.dim_ads dads
	INNER JOIN adw_metrics_glue.dim_campaigns dcamp ON dads.dim_campaign_id = dcamp.dim_campaign_id
	INNER JOIN adw_metrics_glue.dim_advertisers dadv ON dcamp.dim_advertiser_id = dadv.dim_advertiser_id
    INNER JOIN adw_metrics_glue.dim_ad_product_types dapt on dads.dim_ad_product_type_id = dapt.dim_ad_product_type_id
    INNER JOIN dvde.pvc_advertiser_mapping pam ON dadv.cfid = pam.rodeo_advertiser_id 
                      
      where dcamp.campaign_status = 'RUNNING' -- filter is campaign is running
      AND dads.ad_status = 'RUNNING' -- filter if ads are running
      AND DATEDIFF(day, dads.start_dt_utc, dads.end_dt_utc) > 14 -- filter if start and end date are 14 days apart
      and pam.rodeo_advertiser_id is not null
group by 
    dads.dim_ad_id
)
select
ds.base_seg_id,
dim_date,
ds.marketplace_id,
da.cfid as ad_cfid,
dc.cfid as campaign_cfid,
dc.start_dt_utc,
dc.end_dt_utc,
dc.dim_advertiser_id,
dadv.advertiser_name,
dc.campaign_name,
ae.entity_id,
case when seg_class_code = 'CUSTOM_ASIN' then 'Other' else seg_name end as seg_name,
tm.targ_method,
sum(imp_count) as impressions,
sum(click_count) as clicks
from
adw_metrics_glue.WRT_DAILY_SEG_AD_TRAFFIC dsat
inner join adw_metrics_glue.dim_ads da on dsat.dim_ad_id = da.dim_ad_id
inner join adw_metrics_glue.dim_campaigns dc on da.dim_campaign_id = dc.dim_campaign_id
inner join adw_metrics_glue.dim_ad_product_types dapt on da.dim_ad_product_type_id = dapt.dim_ad_product_type_id
inner join adw_metrics_glue.dim_segments ds ON dsat.dim_seg_id = ds.dim_seg_id
inner join adw_metrics_glue.dim_targeting_methods tm ON dsat.dim_targ_method_id = tm.dim_targ_method_id
inner join adw_metrics_glue.dim_advertisers dadv ON dc.dim_advertiser_id = dadv.dim_advertiser_id
inner join adw_metrics_glue.dim_advertiser_entities ae ON dadv.advertiser_id = ae.advertiser_id and is_deleted = 0
where  dim_date between to_date('{RUN_DATE_YYYYMMDD}','YYYYMMDD')-18 and to_date('{RUN_DATE_YYYYMMDD}','YYYYMMDD')
and dsat.dim_ad_id in (select ad_id from filtered_ads)

group by
ds.base_seg_id,
dim_date,
ds.marketplace_id,
da.cfid,
dc.cfid,
dc.start_dt_utc,
dc.end_dt_utc,
dc.dim_advertiser_id,
dadv.advertiser_name,
dc.campaign_name,
  ae.entity_id,
case when seg_class_code = 'CUSTOM_ASIN' then 'Other' else seg_name end,
tm.targ_method

