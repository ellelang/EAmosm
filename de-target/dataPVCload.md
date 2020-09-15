Test ad_cfid IN (9734794310501, 7557170310401, 5876111200201, 3692412370701, 2352684890701, 3226745300601, 3133083920101, 2230898560801, 7851697750801, 5720813520701, 8594754170501)

```
ad_cfid
3.133084e+12    34
3.226745e+12    28
3.692412e+12    38
7.557170e+12    45
7.851698e+12    19
8.594754e+12    29
Name: base_seg_id, dtype: int64
```

Traffic, Conversion, Performance 

·    https://datanet.amazon.com/J19334213;

·    https://datanet.amazon.com/J19348369

·    https://datanet.amazon.com/J19350267

·    https://datanet.amazon.com/J19321497

·    https://datanet.amazon.com/J19334220

·    https://datanet.amazon.com/J19347295

·    https://datanet.amazon.com/J19350273

·    https://datanet.amazon.com/J19321697

·    https://datanet.amazon.com/J19354227

·    https://datanet.amazon.com/J19354170

·    https://datanet.amazon.com/J19354170

Loading Jobs

·    https://datanet.amazon.com/J19343353

·    https://datanet.amazon.com/J19343350

·    https://datanet.amazon.com/J19356679

 [1] HBO Campaign links: 

Test 1: https://advertising.amazon.com/dsp/ENTITYA6I16E0BHHHY/orders/9006197600701/line-items;

Test 2: https://advertising.amazon.com/dsp/ENTITYA6I16E0BHHHY/orders/8600994610201/line-items ; 

Test 3: https://advertising.amazon.com/dsp/ENTITYA6I16E0BHHHY/orders/1495784680201/line-items

[1] Cinemax Campaign links: 

Test 1: https://advertising.amazon.com/dsp/ENTITYA6I16E0BHHHY/orders/4551929070701/line-items  

Control 1: https://advertising.amazon.com/dsp/ENTITYA6I16E0BHHHY/orders/9930155370301/line-items

Test 2: https://advertising.amazon.com/dsp/ENTITYA6I16E0BHHHY/orders/4260404440401/line-items

Control 2: https://advertising.amazon.com/dsp/ENTITYA6I16E0BHHHY/orders/5393825020801/line-items



advertisers:

| Advertisers                                    |
| ---------------------------------------------- |
| EPIX - Channels - US                           |
| CBS All Access Channels - US                   |
| PBS Distribution for Kids Channel - US         |
| Shudder Channel - US                           |
| Hallmark Channel - US                          |
| Cinemax - US                                   |
| Sundance Now - US                              |
| PBS Distribution for Masterpiece Channels - US |
| Starz Entertainment, LLC                       |
| True Crime Files by ID - Channels - US         |
| Motor Trend - US                               |
| Gaia Channel - US (prev Gaiam)                 |
| Showtime - US                                  |
| HBO - Television - US                          |



```SQL
DROP TABLE IF EXISTS vap_agg.games_conversion2019 CASCADE;
CREATE TABLE vap_agg.games_conversion2019
( 
dim_date              timestamp,
adv_country        varchar(64)
adv_cfid                bigint,
adv_name           varchar(500),
camp_cfid          bigint,
camp_name          varchar(2056),
camp_start          timestamp,
camp_end             timestamp,
line_item              bigint,
ad_name            varchar(2056),
ad_start        timestamp,
ad_end               timestamp,
click_purchases            bigint,
view_purchases              bigint,
click_dpv           bigint,
view_dpv            bigint
);

grant select on table vap_agg.games_conversion2019 to device_advertising_rs_etl ;
grant delete on table vap_agg.games_conversion2019 to device_advertising_rs_etl ;
grant insert on table vap_agg.games_conversion2019 to device_advertising_rs_etl ;
grant update on table vap_agg.games_conversion2019 to device_advertising_rs_etl ;

COMMIT;
```









```sql
DROP TABLE IF EXISTS vap_agg.seg_pvc_performance CASCADE;
CREATE TABLE vap_agg.seg_pvc_performance
(
  base_seg_id          varchar(64), 
dim_date              timestamp,
marketplace_id         numeric(38,4),
ad_cfid                bigint,
campaign_cfid           bigint,
start_dt_utc            timestamp,
end_dt_utc              timestamp,
dim_advertiser_id       numeric(38,4),
advertiser_name         varchar(2056),
campaign_name          varchar(2056),
entity_id                varchar(256),
seg_name               varchar(1024),
targ_method              varchar(256),
impressions           numeric(38,4),
clicks                 numeric(38,4),
conversions             numeric(38,4)
);

grant select on table vap_agg.seg_pvc_performance to device_advertising_rs_etl ;
grant delete on table vap_agg.seg_pvc_performance to device_advertising_rs_etl ;
grant insert on table vap_agg.seg_pvc_performance to device_advertising_rs_etl ;
grant update on table vap_agg.seg_pvc_performance to device_advertising_rs_etl ;

COMMIT;
```

```sql
base_seg_id           bigint, 
dim_date              timestamp,
marketplace_id         numeric(38,4),
ad_cfid                bigint,
campaign_cfid           bigint,
start_dt_utc            timestamp,
end_dt_utc              timestamp,
dim_advertiser_id       numeric(38,4),
advertiser_name         varchar(2056),
campaign_name          varchar(2056),
entity_id                varchar(256),
seg_name               varchar(1024),
targ_method              varchar(256),
impressions           numeric(38,4),
clicks                 numeric(38,4),
conversions             numeric(38,4),
```



/dss/dwp/data/pvc_performance2019others_{JOBRUN_DETAILS}.xml



```
1 0:04:21.012591 
2 0:08:34.617464
3 0:13:00.374902
4 0:18:10.467066
5 0:22:35.045965
6 0:26:52.401976
7 0:31:26.155374
8 0:35:56.453274
9 0:40:24.102164
10 0:44:59.144832
11 0:49:38.554685
12 0:54:25.310213
13 0:59:16.871447
14 1:04:04.692995
15 1:08:45.489445
```

|      | imp      | rmse     | overlap  |
| ---- | -------- | -------- | -------- |
| 0    | 4.174902 | 0.004283 | 0.755932 |
| 1    | 4.222032 | 0.004049 | 0.759322 |
| 2    | 4.245753 | 0.004351 | 0.776271 |
| 3    | 4.184019 | 0.004211 | 0.762712 |
| 4    | 4.284895 | 0.004019 | 0.762712 |
| 5    | 4.208791 | 0.004178 | 0.749153 |
| 6    | 4.205932 | 0.003942 | 0.759322 |
| 7    | 4.176591 | 0.003828 | 0.755932 |
| 8    | 4.204629 | 0.004148 | 0.749153 |
| 9    | 4.159304 | 0.004782 | 0.752542 |
| 10   | 4.299581 | 0.004105 | 0.732203 |
| 11   | 4.225696 | 0.004492 | 0.742373 |
| 12   | 4.276984 | 0.004286 | 0.755932 |
| 13   | 4.276839 | 0.004055 | 0.752542 |
| 14   | 4.219701 | 0.004096 | 0.762712 |



HBO - Television - US



| COLUMN_NAME                                    | DATA_TYPE    | PK   | NULLABLE | DEFAULT | AUTOINCREMENT | COMPUTED | REMARKS | JDBC Type | SCALE/SIZE | PRECISION | POSITION |
| ---------------------------------------------- | ------------ | ---- | -------- | ------- | ------------- | -------- | ------- | --------- | ---------- | --------- | -------- |
| base_seg_id                                    | varchar(500) | NO   | YES      |         | NO            | NO       |         | 12        | 500        | 0         | 1        |
| dim_date                                       | timestamp    | NO   | YES      |         | NO            | NO       |         | 93        | 29         | 6         | 2        |
| marketplace_id                                 | bigint       | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 3        |
| ad_cfid                                        | bigint       | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 4        |
| campaign_cfid                                  | bigint       | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 5        |
| start_dt_utc                                   | timestamp    | NO   | YES      |         | NO            | NO       |         | 93        | 29         | 6         | 6        |
| end_dt_utc                                     | timestamp    | NO   | YES      |         | NO            | NO       |         | 93        | 29         | 6         | 7        |
| dim_advertiser_id                              | bigint       | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 8        |
| advertiser_name                                | varchar(500) | NO   | YES      |         | NO            | NO       |         | 12        | 500        | 0         | 9        |
| campaign_name                                  | varchar(500) | NO   | YES      |         | NO            | NO       |         | 12        | 500        | 0         | 10       |
| entity_id                                      | varchar(500) | NO   | YES      |         | NO            | NO       |         | 12        | 500        | 0         | 11       |
| seg_name                                       | varchar(500) | NO   | YES      |         | NO            | NO       |         | 12        | 500        | 0         | 12       |
| targ_method                                    | varchar(500) | NO   | YES      |         | NO            | NO       |         | 12        | 500        | 0         | 13       |
| click_purchases                                | 27196081     | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 14       |
| view_purchases                                 | 1241651385   | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 15       |
| brand_halo_click_purchases                     | 0            | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 16       |
| brand_halo_view_purchases                      | 0            | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 17       |
| click_considerations                           | bigint       | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 18       |
| view_considerations                            | bigint       | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 19       |
| click_pixels                                   | bigint       | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 20       |
| view_pixels                                    | bigint       | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 21       |
| amazon_pay_initial_click_purchases             | 0            | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 22       |
| amazon_pay_initial_view_purchases              | 0            | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 23       |
| amazon_pay_recurring_click_purchases           | 0            | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 24       |
| amazon_pay_recurring_view_purchases            | 0            | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 25       |
| subscription_free_trial_click_purchases        | 6460263      | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 26       |
| subscription_free_trial_view_purchases         | 270881971    | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 27       |
| subscription_initial_click_purchases           | 4501534      | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 28       |
| subscription_initial_view_purchases            | 164179864    | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 29       |
| subscription_recurring_click_purchases         | 437744       | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 30       |
| subscription_recurring_view_purchases          | 50105578     | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 31       |
| subscription_win_back_click_purchases          | 5027057      | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 32       |
| subscription_win_back_view_purchases           | 139322813    | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 33       |
| subscription_initial_promotion_click_purchases | 135851       | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 34       |
| subscription_initial_promotion_view_purchases  | 2181113      | NO   | YES      |         | NO            | NO       |         | -5        | 19         | 0         | 35       |