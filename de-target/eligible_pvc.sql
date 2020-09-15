/*+ ETLM {
    depend:{
       add:[

      {
       name:"adw_metrics_glue.rrt_local_traffic"
      }               
    ]
}
} */
WITH ads_data as
(
SELECT dads.dim_ad_id, dads.cfid, dads.targeting_combined, dads.start_dt_utc, dads.end_dt_utc, dads.dim_budget_id,
    case when dads.targeting_combined like '%imdb%' then 'IMDB'
         when dads.targeting_combined like '%sponsored_content_wizard_featured%' then 'OOBE'
    when dads.targeting_combined like '%sponsored_tiles_horizontal%' then 'Sponsored Tiles'
    when dads.targeting_combined like '%home_fr%' then 'Feature_Rotator'
    when (dads.ad_type = 'CLASS_I_MOBILE_APP' or dads.ad_type = 'CLASS_I_MOBILE_APP_SOV') and dapt.ad_product_type_code in ('DEVICES_CLASS_I_TAKEOVER','DEVICES_CLASS_I', 'AAA_CLASS_II','AAA_CLASS_II','FIRE_TV_BANNER') then 'Fire TV'
    when (dads.ad_type = 'DTCP' or dads.ad_type = 'AAP_MOBILE') and dapt.ad_product_type_code in ('DEVICES_CLASS_I_TAKEOVER','DEVICES_CLASS_I', 'AAA_CLASS_II') then 'Fire Tablet'
    when dapt.ad_product_type_code in ('CLASS_I_MOBILE_WEB','CLASS_I_MOBILE_APP','CLASS_I_DESKTOP','CLASS_I_DESKTOP_SOV','CLASS_I_MOBILE_WEB_SOV') then 'Class 1'
    when dapt.ad_product_type_code in ('AAP_DESKTOP','AAP_MOBILE_WEB','AAP_MOBILE_APP','AAA_MOBILE_NETWORK','AAA_MOBILE_OO','DEVICES_CLASS_II','AAA_FIRE_TV','AAA_KSO') then 'DSP'
    when dapt.ad_product_type_code in ('AAP_VIDEO','CLASS_I_VIDEO_SOV','OTT_VIDEO_1P_GUARANTEED','CLASS_I_VIDEO','OTT_VIDEO_GUARANTEED') then 'Video' else 'Other' end as property_type
FROM 
	adw_metrics_glue.dim_ads dads
	INNER JOIN adw_metrics_glue.dim_campaigns dcamp ON dads.dim_campaign_id = dcamp.dim_campaign_id
	INNER JOIN adw_metrics_glue.dim_advertisers dadv ON dcamp.dim_advertiser_id = dadv.dim_advertiser_id
    INNER JOIN adw_metrics_glue.dim_ad_product_types dapt on dads.dim_ad_product_type_id = dapt.dim_ad_product_type_id
    INNER JOIN dvde.pvc_advertiser_mapping pam ON dadv.cfid = pam.rodeo_advertiser_id 
                      
      --where dcamp.campaign_status = 'RUNNING' -- filter is campaign is running
      --AND dads.ad_status = 'RUNNING' -- filter if ads are running
      WHERE dads.start_dt_utc > to_date('20190630','YYYYMMDD')
      AND dads.end_dt_utc < to_date('20190931','YYYYMMDD')
      AND DATEDIFF(day, dads.start_dt_utc, dads.end_dt_utc) > 14 -- filter if start and end date are 14 days apart
      and pam.rodeo_advertiser_id is not null
group by 
    dads.dim_ad_id,dads.cfid,  dads.targeting_combined, dads.start_dt_utc, dads.end_dt_utc, dads.dim_budget_id,
    case when dads.targeting_combined like '%imdb%' then 'IMDB'
         when dads.targeting_combined like '%sponsored_content_wizard_featured%' then 'OOBE'
    when dads.targeting_combined like '%sponsored_tiles_horizontal%' then 'Sponsored Tiles'
    when dads.targeting_combined like '%home_fr%' then 'Feature_Rotator'
    when (dads.ad_type = 'CLASS_I_MOBILE_APP' or dads.ad_type = 'CLASS_I_MOBILE_APP_SOV') and dapt.ad_product_type_code in ('DEVICES_CLASS_I_TAKEOVER','DEVICES_CLASS_I', 'AAA_CLASS_II','AAA_CLASS_II','FIRE_TV_BANNER') then 'Fire TV'
    when (dads.ad_type = 'DTCP' or dads.ad_type = 'AAP_MOBILE') and dapt.ad_product_type_code in ('DEVICES_CLASS_I_TAKEOVER','DEVICES_CLASS_I', 'AAA_CLASS_II') then 'Fire Tablet'
    when dapt.ad_product_type_code in ('CLASS_I_MOBILE_WEB','CLASS_I_MOBILE_APP','CLASS_I_DESKTOP','CLASS_I_DESKTOP_SOV','CLASS_I_MOBILE_WEB_SOV') then 'Class 1'
    when dapt.ad_product_type_code in ('AAP_DESKTOP','AAP_MOBILE_WEB','AAP_MOBILE_APP','AAA_MOBILE_NETWORK','AAA_MOBILE_OO','DEVICES_CLASS_II','AAA_FIRE_TV','AAA_KSO') then 'DSP'
    when dapt.ad_product_type_code in ('AAP_VIDEO','CLASS_I_VIDEO_SOV','OTT_VIDEO_1P_GUARANTEED','CLASS_I_VIDEO','OTT_VIDEO_GUARANTEED') then 'Video' else 'Other' end
)
,
rrt_data as (
SELECT traf.dim_ad_id, sum(impressions) as total_impressions, sum(revenue_amount)/100 as revenue
     FROM adw_metrics_glue.rrt_local_traffic traf
     JOIN ads_data ad ON traf.dim_Ad_id = ad.dim_ad_id
     GROUP BY traf.dim_Ad_id
)

select  ad.dim_Ad_id,ad.cfid
, property_type
, sum(total_impressions) as total_impressions
, sum(revenue) as revenue 
from ads_data ad
JOIN rrt_data rd ON ad.dim_ad_id = rd.dim_Ad_id
--JOIN budget_filtered b ON ad.dim_ad_id = b.dim_Ad_id
where total_impressions > 100 -- impressions should be greater than 100
and property_type in ('Fire Tablet','Fire TV', 'Class 1','DSP') -- Filter to property type
group by ad.dim_Ad_id,ad.cfid, property_type

  