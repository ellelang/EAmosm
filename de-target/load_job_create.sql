DROP TABLE IF EXISTS vap_agg.eligible_campaigns_pvc CASCADE;
CREATE TABLE vap_agg.eligible_campaigns_pvc
(
   dim_ad_id             bigint,
   cfid                  bigint,
   property_type         varchar(500),
   total_impressions     bigint,
   revenue               numeric(38,4)
);

grant select on table vap_agg.eligible_campaigns_pvc to device_advertising_rs_etl ;
grant delete on table vap_agg.eligible_campaigns_pvc to device_advertising_rs_etl ;
grant insert on table vap_agg.eligible_campaigns_pvc to device_advertising_rs_etl ;
grant update on table vap_agg.eligible_campaigns_pvc to device_advertising_rs_etl ;

COMMIT;