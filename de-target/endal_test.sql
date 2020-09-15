SELECT *
FROM vap_agg.endal_autos_eligible_campaigns
LIMIT 100;

SELECT *
FROM vap_agg.endal_autos_eligible_campaigns
LIMIT 100;


SELECT COUNT(*)
FROM vap_agg.endal_autos_eligible_campaigns
LIMIT 100;

SELECT *
FROM VAP_ENDAL.ENDAL_ELIGIBLE_CAMPAIGNS eec
WHERE optimizationlever = 'SEGMENT_REMOVAL';

SELECT * 
FROM VAP_AGG.ENDAL_PROP_SEGMENTS
LIMIT 100;

SELECT DISTINCT targ_method
FROM VAP_AGG.ENDAL_PROP_SEGMENTS;

SELECT *
FROM VAP_AGG.ENDAL_SEGMENT_REMOVAL_STEP1
LIMIT 100;

SELECT *
FROM VAP_AGG.ENDAL_AUTOS_SEGMENTS_TRAFFIC
LIMIT 100;

SELECT *
FROM VAP_AGG.ENDAL_AUTOS_SEGMENTS_CONV
LIMIT 100;

SELECT COUNT (DISTINCT base_seg_id) 
FROM VAP_AGG.ENDAL_AUTOS_SEGMENTS_TRAFFIC;


SELECT 
BASE_SEG_ID, 
DIM_SEG_ID, 
SEG_ID, 
SEG_NAME, 
SEG_CLASS_CODE, 
SEG_DESC

FROM adw_metrics_glue.dim_segments
     WHERE MARKETPLACE_ID = 1
            AND SEG_DESC != ''
limit 100;

SELECT COUNT(DISTINCT base_seg_id)
FROM adw_metrics_glue.dim_segments

SELECT *
FROM vap_agg.endal_performance
LIMIT 100;

SELECT *
FROM vap_agg.endal_pvc_segments_traffic
LIMIT 100; 


SELECT COUNT(DISTINCT base_seg_id)
FROM vap_agg.endal_pvc_segments_traffic
LIMIT 100; 


SELECT COUNT(DISTINCT base_seg_id)
FROM vap_agg.endal_pvc_segments_conv
LIMIT 100; 

SELECT *
FROM vap_agg.endal_pvc_segments_conv
LIMIT 100; 

SELECT ad_cfid, campaign_name, newdata
FROM  vap_agg.endal_pvc_segments_traffic dpv
LEFT JOIN VAP_ENDAL.ENDAL_ELIGIBLE_CAMPAIGNS eec
ON dpv.ad_cfid = eec.id
LIMIT 100;

SELECT *
FROM vap_agg.endal_performance_traffic
LIMIT 100;


SELECT COUNT(DISTINCT ad_cfid)
FROM vap_agg.endal_pvc_segments_traffic
LIMIT 100; 

SELECT DISTINCT ad_cfid, AVG(clicks) AS avg_clicks, AVG(impressioins) AS avg_imp
FROM vap_agg.endal_pvc_segments_traffic
GROUP BY ad_cfid; 


SELECT *
FROM vap_agg.a_adw_ott_managed_demand ott
INNER JOIN vap_agg.a_adw_ad_segment seg
ON ott.ad_cfid = seg.dim_ad_id
LIMIT 100;


SELECT *
FROM vap_agg.a_adw_ad_segment_all seg
WHERE seg.dim_date BETWEEN TO_DATE('20190101','YYYYMMDD') AND TO_DATE('20191231','YYYYMMDD')
AND seg.targeting_method = 'untargeted'
AND seg.view_installs is not null
AND seg.click_downloads is not null
LIMIT 8000;

SELECT DISTINCT adv_vertical, apps_ott_channels_flag
FROM vap_agg.a_adw_whisps_segment
LIMIT 100; 

SELECT adv_cfid
, camp_cfid
, seg_id
, ad_type
, vertical
, category
, property_type
, service_type
, adv_vertical
, adv_subvertical
, seg_name
, sum(impressions) as impression
, sum(clicks) as clicks
, sum(revenue) as cost
FROM vap_agg.a_adw_whisps_segment
WHERE apps_ott_channels_flag = 'OTT'
GROUP BY  adv_cfid
, camp_cfid
, seg_id
, ad_type
, vertical
, category
, property_type
, service_type
, adv_vertical
, adv_subvertical
, seg_name
LIMIT 500; 

SELECT *
FROM vap_agg.endal_segment_removal_step1
LIMIT 100; 

SELECT *
FROM vap_agg.a_adw_pvc_ott_app_metrics
LIMIT 100; 

SELECT
pvc.LOCALE AS ADVERTISER_COUNTRY
,adv.CFID AS ADVERTISER_CFID
,pvc.ADVERTISER AS ADVERTISER_NAME
,camp.CFID AS CAMPAIGN_CFID
,pvc.ORDER AS CAMPAIGN_NAME
,camp.START_DT_UTC AS CAMPAIGN_START_DATE
,camp.END_DT_UTC AS CAMPAIGN_END_DATE
,ds.SEG_ID
,ds.SEG_NAME
,SUM(NVL(pvc.TOTAL_COST,0)) AS TOTAL_COST
,SUM(NVL(pvc.TOTAL_COST_RECONCILED,0)) AS TOTAL_COST_RECONCILED
,SUM(NVL(pvc.SUBSCRIPTIONS_PVC_FREE_TRIALS,0))+SUM(NVL(pvc.SUBSCRIPTIONS_PVC_HARD_OFFERS,0)) AS TOTAL_SIGN_UPS
,SUM(NVL(pvc.APPSTORE_APP_DOWNLOADS,0)) AS TOTAL_DOWNLOADS

FROM VAP_AGG.A_ADW_PVC_OTT_APP_METRICS pvc
INNER JOIN ADW_METRICS_GLUE.DIM_CAMPAIGNS camp ON pvc.ORDER_CFID = camp.CFID
INNER JOIN ADW_METRICS_GLUE.DIM_ADVERTISERS adv ON camp.DIM_ADVERTISER_ID = adv.DIM_ADVERTISER_ID
INNER JOIN ADW_METRICS_GLUE.DIM_ADS da ON pvc.LINE_ITEM_CFID = da.CFID
LEFT JOIN ADW_METRICS_GLUE.WRT_SEGMENTS_CONV fdac ON da.DIM_AD_ID = fdac.DIM_AD_ID
INNER JOIN ADW_METRICS_GLUE.DIM_SEGMENTS ds ON ds.DIM_SEG_ID = fdac.DIM_SEG_ID

WHERE
pvc.LOCALE = 'US'
AND pvc.PVCMETRICS_CLOSEDBETA = 1
AND pvc.DATE BETWEEN TO_DATE('20190701','YYYYMMDD') AND TO_DATE('20190731','YYYYMMDD')

GROUP BY
pvc.LOCALE 
,adv.CFID 
,pvc.ADVERTISER 
,camp.CFID 
,pvc.ORDER 
,camp.START_DT_UTC
,camp.END_DT_UTC 
,ds.SEG_ID
,ds.SEG_NAME



SELECT pvc.*
,camp.START_DT_UTC AS CAMPAIGN_START_DATE
,camp.END_DT_UTC AS CAMPAIGN_END_DATE
,ds.SEG_ID
,ds.SEG_NAME
FROM VAP_AGG.A_ADW_PVC_OTT_APP_METRICS pvc
INNER JOIN ADW_METRICS_GLUE.DIM_CAMPAIGNS camp ON pvc.ORDER_CFID = camp.CFID
INNER JOIN ADW_METRICS_GLUE.DIM_ADVERTISERS adv ON camp.DIM_ADVERTISER_ID = adv.DIM_ADVERTISER_ID
INNER JOIN ADW_METRICS_GLUE.DIM_ADS da ON pvc.LINE_ITEM_CFID = da.CFID
LEFT JOIN ADW_METRICS_GLUE.WRT_SEGMENTS_CONV fdac ON da.DIM_AD_ID = fdac.DIM_AD_ID
INNER JOIN ADW_METRICS_GLUE.DIM_SEGMENTS ds ON ds.DIM_SEG_ID = fdac.DIM_SEG_ID
WHERE pvc.DATE BETWEEN TO_DATE('20190101','YYYYMMDD') AND TO_DATE('20191231','YYYYMMDD')
LIMIT 100;


/*+ ETLM {  
	depend:{  
		add:[  
			{name:"VAP_AGG.A_ADW_PVC_OTT_APP_METRICS"} 
			,{name:"ADW_METRICS_GLUE.DIM_CAMPAIGNS"} 
			,{name:"ADW_METRICS_GLUE.DIM_ADVERTISERS"}
			,{name:"ADW_METRICS_GLUE.DIM_ADS"}
               ,{name:"VAP_AGG.AD_ASINS"}
               ,{name:"ADW_METRICS_GLUE.DIM_ASINS"}
			,{name:"ADW_METRICS_GLUE.WRT_SEGMENTS_CONV"}
			,{name:"ADW_METRICS_GLUE.DIM_SEGMENTS"}
		]  
	}   
}*/


SELECT
pvc.LOCALE AS ADVERTISER_COUNTRY
,adv.CFID AS ADVERTISER_CFID
,dasin2.asin AS ASINKEY
,dasin2.item_name AS ITEM
,dasin2.merchant_brand_name AS BRAND
,dasin2.dim_gl_product_group_id AS PRODUCT
,pvc.ADVERTISER AS ADVERTISER_NAME
,camp.CFID AS CAMPAIGN_CFID
,pvc.ORDER AS CAMPAIGN_NAME
,camp.START_DT_UTC AS CAMPAIGN_START_DATE
,camp.END_DT_UTC AS CAMPAIGN_END_DATE
,ds.SEG_ID
,ds.SEG_NAME
,SUM(NVL(pvc.IMPRESSIONS,0)) AS TOTAL_IMPRESSIONS
,SUM(NVL(pvc.VIEWABLE_IMPRESSIONS,0)) AS TOTAL_VIEWABLE_IMPRESSIONS
,SUM(NVL(pvc.CLICK_THROUGHS,0)) AS TOTAL_CLICK_THROUGH
,SUM(NVL(pvc.DPV,0)) AS TOTAL_DPV
,SUM(NVL(pvc.PURCHASES,0)) AS TOTAL_PURCHASES
,SUM(NVL(pvc.TOTAL_COST,0)) AS TOTAL_COST
,SUM(NVL(pvc.TOTAL_COST_RECONCILED,0)) AS TOTAL_COST_RECONCILED
,SUM(NVL(pvc.SUBSCRIPTIONS_PVC_FREE_TRIALS,0)) AS TOTAL_PVC_FREE_SIGN
,SUM(NVL(pvc.SUBSCRIPTIONS_PVC_HARD_OFFERS,0)) AS TOTAL_PVC_HARD_SIGN
,SUM(NVL(pvc.SUBSCRIPTIONS_APP_FREE_TRIALS,0)) AS TOTAL_APP_FREE_SIGN
,SUM(NVL(pvc.SUBSCRIPTIONS_APP_HARD_OFFERS,0)) AS TOTAL_APP_HARD_SIGN
,SUM(NVL(pvc.APPSTORE_APP_DOWNLOADS,0)) AS TOTAL_APP_DOWNLOADS

FROM VAP_AGG.A_ADW_PVC_OTT_APP_METRICS pvc
INNER JOIN ADW_METRICS_GLUE.DIM_CAMPAIGNS camp ON pvc.ORDER_CFID = camp.CFID
INNER JOIN ADW_METRICS_GLUE.DIM_ADVERTISERS adv ON camp.DIM_ADVERTISER_ID = adv.DIM_ADVERTISER_ID
INNER JOIN ADW_METRICS_GLUE.DIM_ADS da ON pvc.LINE_ITEM_CFID = da.CFID
INNER JOIN VAP_AGG.AD_ASINS dasin ON da.CFID = dasin.dim_ad_id
LEFT JOIN ADW_METRICS_GLUE.DIM_ASINS dasin2 ON  dasin.dim_asin = dasin2.asin
LEFT JOIN ADW_METRICS_GLUE.WRT_SEGMENTS_CONV fdac ON da.DIM_AD_ID = fdac.DIM_AD_ID
INNER JOIN ADW_METRICS_GLUE.DIM_SEGMENTS ds ON ds.DIM_SEG_ID = fdac.DIM_SEG_ID

WHERE
pvc.LOCALE = 'US'
AND pvc.PVCMETRICS_CLOSEDBETA = 1
AND pvc.DATE BETWEEN TO_DATE('20190701','YYYYMMDD') AND TO_DATE('20190702','YYYYMMDD')
AND pvc.ADVERTISER  = 'EPIX - Channels - US'

GROUP BY
pvc.LOCALE 
,adv.CFID 
,dasin2.asin
,dasin2.item_name
,dasin2.dim_gl_product_group_id
,dasin2.merchant_brand_name
,pvc.ADVERTISER 
,camp.CFID 
,pvc.ORDER 
,camp.START_DT_UTC
,camp.END_DT_UTC 
,ds.SEG_ID
,ds.SEG_NAME;

limit 100;


select *
from VAP_AGG.AD_ASINS

select *
from BOOKER_EXT.D_ACTIVE_ASIN_ATTRIBUTES
LIMIT 100;

select *
from ADW_METRICS_GLUE.DIM_ASINS
limit 100;


SELECT seg.*, b2.asin, b2.item_name, b2.dim_gl_product_group_id, b2.merchant_brand_name
FROM vap_agg.a_adw_ad_segment_all seg
LEFT JOIN VAP_AGG.AD_ASINS dasin ON seg.dim_ad_id = dasin.dim_ad_id
LEFT JOIN ADW_METRICS_GLUE.DIM_ASINS b2 ON dasin.dim_asin = b2.asin
LIMIT 100;


SELECT seg.*, b2.asin, b2.item_name, b2.dim_gl_product_group_id, b2.merchant_brand_name
FROM vap_agg.a_adw_ad_segment_all seg
LEFT JOIN VAP_AGG.AD_ASINS dasin ON seg.dim_ad_id = dasin.dim_ad_id
LEFT JOIN ADW_METRICS_GLUE.DIM_ASINS b2 ON dasin.dim_asin = b2.asin
WHERE seg.dim_date BETWEEN TO_DATE('20190101','YYYYMMDD') AND TO_DATE('20191231','YYYYMMDD')
AND seg.targeting_method = 'untargeted'

---AND seg.view_installs is not null---
AND seg.click_downloads is not null
LIMIT 8000;


SELECT *
from ADW_METRICS_GLUE.DIM_ADS da
where da.dim_ad_id = 1470150547
LIMIT 100;

select a.*
from VAP_AGG.AD_ASINS a
inner join ADW_METRICS_GLUE.DIM_ASINS b2 ON a.dim_asin = b2.asin
WHERE a.dim_ad_id = 1470150547
limit 10;


/*+ ETLM {  
	depend:{  
		add:[  
			{name:"VAP_AGG.A_ADW_PVC_OTT_APP_METRICS"} 
			,{name:"ADW_METRICS_GLUE.DIM_CAMPAIGNS"} 
			,{name:"ADW_METRICS_GLUE.DIM_ADVERTISERS"}
			,{name:"ADW_METRICS_GLUE.DIM_ADS"}
			,{name:"ADW_METRICS_GLUE.WRT_SEGMENTS_CONV"}
			,{name:"ADW_METRICS_GLUE.DIM_SEGMENTS"}
		]  
	}   
}*/


SELECT
pvc.LOCALE AS ADVERTISER_COUNTRY
,adv.CFID AS ADVERTISER_CFID
,pvc.ADVERTISER AS ADVERTISER_NAME
,camp.CFID AS CAMPAIGN_CFID
,pvc.ORDER AS CAMPAIGN_NAME
,camp.START_DT_UTC AS CAMPAIGN_START_DATE
,camp.END_DT_UTC AS CAMPAIGN_END_DATE
,ds.SEG_ID
,ds.SEG_NAME
,SUM(NVL(pvc.TOTAL_COST,0)) AS TOTAL_COST
,SUM(NVL(pvc.TOTAL_COST_RECONCILED,0)) AS TOTAL_COST_RECONCILED
,SUM(NVL(pvc.SUBSCRIPTIONS_PVC_FREE_TRIALS,0))+SUM(NVL(pvc.SUBSCRIPTIONS_PVC_HARD_OFFERS,0)) AS TOTAL_SIGN_UPS
,SUM(NVL(pvc.APPSTORE_APP_DOWNLOADS,0)) AS TOTAL_DOWNLOADS

FROM VAP_AGG.A_ADW_PVC_OTT_APP_METRICS pvc
INNER JOIN ADW_METRICS_GLUE.DIM_CAMPAIGNS camp ON pvc.ORDER_CFID = camp.CFID
INNER JOIN ADW_METRICS_GLUE.DIM_ADVERTISERS adv ON camp.DIM_ADVERTISER_ID = adv.DIM_ADVERTISER_ID
INNER JOIN ADW_METRICS_GLUE.DIM_ADS da ON pvc.LINE_ITEM_CFID = da.CFID
LEFT JOIN ADW_METRICS_GLUE.WRT_SEGMENTS_CONV fdac ON da.DIM_AD_ID = fdac.DIM_AD_ID
INNER JOIN ADW_METRICS_GLUE.DIM_SEGMENTS ds ON ds.DIM_SEG_ID = fdac.DIM_SEG_ID

WHERE
pvc.LOCALE = 'US'
AND pvc.PVCMETRICS_CLOSEDBETA = 1
AND pvc.DATE BETWEEN TO_DATE('20190701','YYYYMMDD') AND TO_DATE('20190731','YYYYMMDD')

GROUP BY
pvc.LOCALE 
,adv.CFID 
,pvc.ADVERTISER 
,camp.CFID 
,pvc.ORDER 
,camp.START_DT_UTC
,camp.END_DT_UTC 
,ds.SEG_ID
,ds.SEG_NAME


SELECT *
FROM vap_agg.a_adw_app_download_metrics
LIMIT 100;


SELECT *
FROM ADW_METRICS_GLUE.WRT_SEGMENTS_CONV seg
INNER JOIN vap_agg.a_adw_app_download_metrics app ON seg.dim_ad_id = app.dim_ad_id
WHERE app.advertiser_country = 'US'
AND app.advertiser_cfid = 4778490886216
AND app.dim_date BETWEEN TO_DATE('20190701','YYYYMMDD') AND TO_DATE('20190731','YYYYMMDD')
limit 100;

SELECT 
pvc.date
,pvc.advertiser
,pvc.order_cfid
,pvc.line_item_cfid
,da.targeting
,da.ad_type
,dasin.dim_asin
,SUM(NVL(pvc.IMPRESSIONS,0)) AS TOTAL_IMPRESSIONS
FROM VAP_AGG.A_ADW_PVC_OTT_APP_METRICS pvc 
INNER JOIN ADW_METRICS_GLUE.DIM_ADS da ON pvc.LINE_ITEM_CFID = da.CFID
LEFT JOIN VAP_AGG.AD_ASINS dasin ON da.CFID = dasin.dim_ad_id
WHERE pvc.locale = 'US'
GROUP BY 
pvc.date
,pvc.advertiser
,pvc.order_cfid
,pvc.line_item_cfid
,da.targeting
,da.ad_type
,dasin.dim_asin
ORDER BY pvc.date desc
limit 100; 


SELECT DISTINCT da.optimization_type
FROM ADW_METRICS_GLUE.DIM_ADS da 
LIMIT 100;

SELECT *
FROM VAP_AGG.ENDAL_PVC_SEGMENTS_CONV ecv
INNER JOIN VAP_AGG.AD_ASINS dasin ON ecv.ad_cfid = dasin.dim_ad_id
LIMIT 100;

SELECT advertiser_sub_industry, COUNT(*)
FROM VAP_AGG.ENDAL_PVC_SEGMENTS_CONV ecv
INNER JOIN ADW_METRICS_GLUE.DIM_ADVERTISERS dav ON ecv.dim_advertiser_id = dav.dim_advertiser_id
GROUP BY advertiser_sub_industry
LIMIT 100;

SELECT ecv*, dav.advertiser_sub_industry, trf.*
FROM VAP_AGG.ENDAL_PVC_SEGMENTS_CONV ecv
INNER JOIN ADW_METRICS_GLUE.DIM_ADVERTISERS dav ON ecv.dim_advertiser_id = dav.dim_advertiser_id
INNER JOIN VAP_AGG.ENDAL_PVC_SEGMENTS_TRAFFIC trf ON ecv.base_seg_id = trf.base_seg_id
AND ecv.ad_cfid= trf.ad_cfid AND ecv.campaign_id = trf.campaign_id
WHERE ecv.dim_date BETWEEN TO_DATE('20190101','YYYYMMDD') AND TO_DATE('20191231','YYYYMMDD')
LIMIT 100;

WITH CTE1 AS
(SELECT *
FROM VAP_AGG.ENDAL_PVC_SEGMENTS_CONV ecv
INNER JOIN VAP_AGG.ENDAL_PVC_SEGMENTS_TRAFFIC trf ON ecv.base_seg_id = trf.base_seg_id 
AND ecv.ad_cfid= trf.ad_cfid)

SELECT *, dav.advertiser_sub_industry,
FROM CTE1
LEFT JOIN ADW_METRICS_GLUE.DIM_ADVERTISERS dav ON CTE1.dim_advertiser_id = dav.dim_advertiser_id
--WHERE ecv.dim_date BETWEEN TO_DATE('20190101','YYYYMMDD') AND TO_DATE('20191231','YYYYMMDD')
LIMIT 100;

WITH cte1 as(SELECT *
FROM VAP_AGG.ENDAL_PVC_SEGMENTS_CONV ecv
LEFT JOIN VAP_AGG.ENDAL_PVC_SEGMENTS_TRAFFIC trf ON ecv.base_seg_id = trf.base_seg_id AND ecv.ad_cfid= trf.ad_cfid AND 
ecv.campaign_cfid = trf.campaign_cfid AND ecv.start_dt_utc = trf.start_dt_utc AND ecv.end_dt_utc = trf.end_dt_utc 
LEFT JOIN ADW_METRICS_GLUE.DIM_ADVERTISERS dav ON ecv.dim_advertiser_id = dav.dim_advertiser_id)

SELECT *
FROM cte1
LIMIT 10;


WITH CTE1 AS (SELECT 
pvc.date
,pvc.advertiser
,pvc.order_cfid
,pvc.line_item_cfid
,da.targeting
,da.ad_type
,dasin.dim_asin
,SUM(NVL(pvc.IMPRESSIONS,0)) AS TOTAL_IMPRESSIONS
,SUM(NVL(pvc.TOTAL_COST,0)) AS TOTAL_COST
,SUM(NVL(pvc.SUBSCRIPTIONS_PVC_FREE_TRIALS,0))+SUM(NVL(pvc.SUBSCRIPTIONS_PVC_HARD_OFFERS,0)) AS TOTAL_PVC_SIGN_UPS
,SUM(NVL(pvc.SUBSCRIPTIONS_APP_FREE_TRIALS,0)) + SUM(NVL(pvc.SUBSCRIPTIONS_APP_HARD_OFFERS,0)) AS TOTAL_APP__SIGN_UPS
,SUM(NVL(pvc.APPSTORE_APP_DOWNLOADS,0)) AS TOTAL_APP_DOWNLOADS
FROM VAP_AGG.A_ADW_PVC_OTT_APP_METRICS pvc 
INNER JOIN ADW_METRICS_GLUE.DIM_ADS da ON pvc.LINE_ITEM_CFID = da.CFID
LEFT JOIN VAP_AGG.AD_ASINS dasin ON da.CFID = dasin.dim_ad_id

WHERE
pvc.LOCALE = 'US'
AND pvc.PVCMETRICS_CLOSEDBETA = 1
AND pvc.DATE BETWEEN TO_DATE('20190101','YYYYMMDD') AND TO_DATE('20191231','YYYYMMDD')
--AND pvc.advertiser = 'CBS All Access Channels - US'

GROUP BY 
pvc.date
,pvc.advertiser
,pvc.order_cfid
,pvc.line_item_cfid
,da.targeting
,da.ad_type
,dasin.dim_asin
ORDER BY pvc.date desc)

SELECT COUNT(*)
FROM CTE1
WHERE TOTAL_IMPRESSIONS >=100
limit 100; 


SELECT ecv.*, trf.impressions, trf.clicks,
FROM vap_agg.segment_pvc_conversion ecv
INNER JOIN vap_agg.segment_pvc_traffic trf ON ecv.base_seg_id = trf.base_seg_id 
AND ecv.ad_cfid= trf.ad_cfid
AND ecv.campaign_cfid = trf.campaign_cfid 
AND ecv.start_dt_utc = trf.start_dt_utc 
AND ecv.end_dt_utc = trf.end_dt_utc 
WHERE ecv.targ_method = 'untargeted'
AND ecv.advertiser_name IN ('HBO - Television - US', 'Starz Entertainment, LLC')
AND ecv.marketplace_id = 1
AND trf.impressions >= 100
LIMIT 100;


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
VAP_AGG.SEGMENT_PVC_TRAFFIC st
LEFT JOIN VAP_AGG.SEGMENT_PVC_CONVERSION sc
ON st.base_seg_id = sc.base_seg_id
and st.dim_date= sc.dim_Date
and st.marketplace_id=sc.marketplace_id
and st.ad_cfid = sc.ad_cfid
and st.targ_method = sc.targ_method
and st.seg_name = sc.seg_name
where st.targ_method = 'untargeted'  -- Filter on only untargeted segments
and st.advertiser_name IN ('HBO - Television - US', 'Starz Entertainment, LLC')
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
);



case when sum(conversions) > 0 
then nvl(((sum(st.impressions)/1000)*(max(a.revenue)/(max(a.total_impressions)/1000)))/sum(conversions),0) 
else 0 end as cost_per_conversion,
