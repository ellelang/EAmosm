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
WHERE dadv.dim_advertiser_id in ('112097',  -- Acura       Filter advertiser list
                            '384635878', -- Audi
                            '86065',  -- BMW
                          --  '69185', -- Castrol
                          --  '21772', -- CDK
                          --  '155846', --Chevron
                            '13001', --Chrysler
                          --  '751312008', --Cooper Tires
                          --  '21831284', --Edmunds
                            '53187', --Fiat
                            '17470', --Ford
                            '1405',  --General Motors
                            '2227984', --Harley-Davidson
                            '9663', --Honda
                          '25157', --Hyundai
                            '18971', --Infiniti
                            '224637', --Jaguar
                          '29742', --Kia
                            '72945', --Land Rover
                            '17471', --Lexus
                            '47488', --Lincoln
                          --  '82001964', --Mazda
                            '21042', --Mercedes-Benz
                          --  '16627', --Michelin
                            '31728733', --Mini
                            '99021708', --Mitsubishi
                            '18438', --Nissan Motor
                            '732652849', --Porsche
                          --  '18347', --Shell
                            '47497', --Subaru
                           -- '17533' --Toyota
                          --  '61846595', --Valvoline
                            '116006', --Volkswagen
                            '32777' -- Volvo
                          --  '49005' --Yokohama   
                                )                        
      AND dcamp.campaign_status = 'RUNNING' -- filter is campaign is running
      AND dads.ad_status = 'RUNNING' -- filter if ads are running
      AND DATEDIFF(day, dads.start_dt_utc, dads.end_dt_utc) > 14 -- filter if start and end date are 14 days apart
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
),
rrt_data as (
SELECT traf.dim_ad_id, sum(impressions) as total_impressions, sum(revenue_amount)/100 as revenue
     FROM adw_metrics_glue.rrt_local_traffic traf
     JOIN ads_data ad ON traf.dim_Ad_id = ad.dim_ad_id
     GROUP BY traf.dim_ad_id
),
budget as (
select dim_ad_id,dim_budget_id,spend_amount, budget_amount, start_dt_utc, end_dt_utc, budget_impressions, sum(impressions) over (partition by dim_campaign_id) as campaign_impressions from (
select ad.dim_ad_id,  ad.dim_budget_id, CAST(sum(revenue_amount) as decimal) as spend_amount, dim_campaign_id,
         dbud.budget_amount  as budget_amount, ad.start_dt_utc, ad.end_dt_utc, budget_impressions, sum(traf.impressions) as impressions
  from adw_metrics_glue.rrt_local_traffic traf
      JOIN ads_data ad ON traf.dim_Ad_id = ad.dim_ad_id
      INNER JOIN adw_metrics_glue.dim_budgets dbud ON dbud.dim_budget_id = ad.dim_budget_id

  GROUP BY ad.dim_ad_id,dim_campaign_id,
           ad.dim_budget_id,
           dbud.budget_amount,
  		   ad.start_dt_utc,
           ad.end_dt_utc,
           dbud.budget_impressions

)),
budget_filtered as 
  (select distinct dim_Ad_id from (
select dim_Ad_id, case when budget_amount is not null then (spend_amount/1000) else campaign_impressions end as actual, case when budget_amount is not null then budget_Amount else budget_impressions end as budget from (
select fads2.dim_ad_id, fads2.spend_amount * 1000 / fads2.budget_amount, CAST(DATEDIFF(day, start_dt_utc, GETDATE()) as DECIMAL) / CAST(DATEDIFF(day, start_dt_utc, end_dt_utc) as DECIMAL), fads2.spend_Amount, ((fads2.budget_amount)*CAST(DATEDIFF(day, start_dt_utc, GETDATE()) as DECIMAL))/CAST(DATEDIFF(day, start_dt_utc, end_dt_utc) as DECIMAL), CAST(DATEDIFF(day, start_dt_utc, GETDATE()) as DECIMAL), CAST(DATEDIFF(day, start_dt_utc, end_dt_utc) as DECIMAL) as budget_amount,
(budget_impressions*CAST(DATEDIFF(day, start_dt_utc, GETDATE()) as DECIMAL))/CAST(DATEDIFF(day, start_dt_utc, end_dt_utc) as DECIMAL) as budget_impressions, campaign_impressions
 from budget fads2
))
where actual>=budget*.9 -- delivery filter criteria
  )
select ad.dim_Ad_id,ad.cfid, case when left(targeting_combined,24)= '((geo=US OR geoip=US)AND' then substring(targeting_combined,26,len(targeting_combined)-(26+1)) else substring(targeting_combined,3,len(targeting_combined)-(3+1)) end as targeting_combined, property_type, sum(total_impressions) as total_impressions, sum(revenue) as revenue from 
ads_data ad
JOIN rrt_data rd ON ad.dim_ad_id = rd.dim_Ad_id
JOIN budget_filtered b ON ad.dim_ad_id = b.dim_Ad_id
where total_impressions > 100 -- impressions should be greater than 100
and property_type in ('Fire Tablet','Fire TV', 'Class 1','DSP') -- Filter to property type
group by ad.dim_Ad_id,ad.cfid, case when left(targeting_combined,24)= '((geo=US OR geoip=US)AND' then substring(targeting_combined,26,len(targeting_combined)-(26+1)) else substring(targeting_combined,3,len(targeting_combined)-(3+1)) end, property_type
  