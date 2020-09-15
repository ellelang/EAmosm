/*+ ETLM {
    depend:{
       add:[

      {
       name:"adw_metrics_glue.WRT_SEGMENTS_CONV",
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
    SUM(CASE WHEN dsat.dim_conv_type_id in (1,2) AND dsat.dim_engagement_scope_id = 1
        THEN dsat.click_conversion_count ELSE 0 END) AS click_purchases,
    SUM(CASE WHEN dsat.dim_conv_type_id in (1,2) AND dsat.dim_engagement_scope_id = 1
        THEN dsat.impression_conversion_count ELSE 0 END) AS view_purchases,
    SUM(CASE WHEN dsat.dim_conv_type_id in (1,2) AND dsat.dim_engagement_scope_id = 2
        THEN dsat.click_conversion_count ELSE 0 END) AS brand_halo_click_purchases,
    SUM(CASE WHEN dsat.dim_conv_type_id in (1,2) AND dsat.dim_engagement_scope_id = 2
        THEN dsat.impression_conversion_count ELSE 0 END) AS brand_halo_view_purchases,
    SUM(CASE WHEN dsat.dim_conv_type_id = 3 AND dsat.dim_engagement_scope_id = 1
        THEN dsat.click_conversion_count ELSE 0 END) AS click_considerations,
    SUM(CASE WHEN dsat.dim_conv_type_id = 3 AND dsat.dim_engagement_scope_id = 1
        THEN dsat.impression_conversion_count ELSE 0 END) AS view_considerations,
    SUM(CASE WHEN dsat.dim_conv_type_id in (
            4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,29,30,31,32,36,37,43,44,45,46,47,48,49,53,54,55,56,57,58,59,60,61,62,63,64,80,81,82,83)
        AND dsat.dim_engagement_scope_id = 1
        THEN dsat.click_conversion_count ELSE 0 END) AS click_pixels,
    SUM(CASE WHEN dsat.dim_conv_type_id in (
            4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,29,30,31,32,36,37,43,44,45,46,47,48,49,53,54,55,56,57,58,59,60,61,62,63,64,80,81,82,83)
        AND dsat.dim_engagement_scope_id = 1
        THEN dsat.impression_conversion_count ELSE 0 END) AS view_pixels,
    SUM(CASE WHEN dsat.dim_conv_type_id = 115 AND dsat.dim_engagement_scope_id = 1
        THEN dsat.click_conversion_count ELSE 0 END) AS amazon_pay_initial_click_purchases,
    SUM(CASE WHEN dsat.dim_conv_type_id = 115 AND dsat.dim_engagement_scope_id = 1
        THEN dsat.impression_conversion_count ELSE 0 END) AS amazon_pay_initial_view_purchases,
    SUM(CASE WHEN dsat.dim_conv_type_id = 116 AND dsat.dim_engagement_scope_id = 1
        THEN dsat.click_conversion_count ELSE 0 END) AS amazon_pay_recurring_click_purchases,
    SUM(CASE WHEN dsat.dim_conv_type_id = 116 AND dsat.dim_engagement_scope_id = 1
        THEN dsat.impression_conversion_count ELSE 0 END) AS amazon_pay_recurring_view_purchases,
    SUM(CASE WHEN dsat.dim_conv_type_id = 156 AND dsat.dim_engagement_scope_id = 1
        THEN dsat.click_conversion_count ELSE 0 END) AS subscription_free_trial_click_purchases,
    SUM(CASE WHEN dsat.dim_conv_type_id = 156 AND dsat.dim_engagement_scope_id = 1
        THEN dsat.impression_conversion_count ELSE 0 END) AS subscription_free_trial_view_purchases,
    SUM(CASE WHEN dsat.dim_conv_type_id = 157 AND dsat.dim_engagement_scope_id = 1
        THEN dsat.click_conversion_count ELSE 0 END) AS subscription_initial_click_purchases,
    SUM(CASE WHEN dsat.dim_conv_type_id = 157 AND dsat.dim_engagement_scope_id = 1
        THEN dsat.impression_conversion_count ELSE 0 END) AS subscription_initial_view_purchases,
    SUM(CASE WHEN dsat.dim_conv_type_id = 158 AND dsat.dim_engagement_scope_id = 1
        THEN dsat.click_conversion_count ELSE 0 END) AS subscription_recurring_click_purchases,
    SUM(CASE WHEN dsat.dim_conv_type_id = 158 AND dsat.dim_engagement_scope_id = 1
        THEN dsat.impression_conversion_count ELSE 0 END) AS subscription_recurring_view_purchases,
    SUM(CASE WHEN dsat.dim_conv_type_id = 159 AND dsat.dim_engagement_scope_id = 1
        THEN dsat.click_conversion_count ELSE 0 END) AS subscription_win_back_click_purchases,
    SUM(CASE WHEN dsat.dim_conv_type_id = 159 AND dsat.dim_engagement_scope_id = 1
        THEN dsat.impression_conversion_count ELSE 0 END) AS subscription_win_back_view_purchases,
    SUM(CASE WHEN dsat.dim_conv_type_id = 160 AND dsat.dim_engagement_scope_id = 1
        THEN dsat.click_conversion_count ELSE 0 END) AS subscription_initial_promotion_click_purchases,
    SUM(CASE WHEN dsat.dim_conv_type_id = 160 AND dsat.dim_engagement_scope_id = 1
        THEN dsat.impression_conversion_count ELSE 0 END) AS subscription_initial_promotion_view_purchases

from
adw_metrics_glue.WRT_SEGMENTS_CONV dsat
inner join adw_metrics_glue.dim_ads da on dsat.dim_ad_id = da.dim_ad_id
inner join adw_metrics_glue.dim_campaigns dc on da.dim_campaign_id = dc.dim_campaign_id
inner join adw_metrics_glue.dim_ad_product_types dapt on da.dim_ad_product_type_id = dapt.dim_ad_product_type_id
inner join adw_metrics_glue.dim_segments ds ON dsat.dim_seg_id = ds.dim_seg_id
inner join adw_metrics_glue.dim_targeting_methods tm ON dsat.dim_targ_method_id = tm.dim_targ_method_id
inner join adw_metrics_glue.dim_advertisers dadv ON dc.dim_advertiser_id = dadv.dim_advertiser_id
inner join adw_metrics_glue.dim_advertiser_entities ae ON dadv.advertiser_id = ae.advertiser_id and is_deleted = 0
where  dim_date between to_date('{RUN_DATE_YYYYMMDD}','YYYYMMDD')-18 and to_date('{RUN_DATE_YYYYMMDD}','YYYYMMDD')-4
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