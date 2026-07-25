


# Start the database

```sh
# start
docker compose up -d

# check log
docker logs -f hara_chicken_timescaledb

# connect
docker compose exec timescaledb psql -U hara -d chicken_tracking
```

Verify DB:
```sql
-- verify extensions:
SELECT extname, extversion
FROM pg_extension
WHERE extname IN ('timescaledb', 'postgis');

extname   | extversion 
-------------+------------
 postgis     | 3.6.3
 timescaledb | 2.27.2
          
          
          
--Verify hypertable:
SELECT hypertable_name
FROM timescaledb_information.hypertables;

hypertable_name
-----------------
tracking_points
```

# SQL operation

## Insert one chicken tracking point

```sql
INSERT INTO tracking_points (
    ts,
    video_id,
    pen_id,
    frame_index,
    track_id,
    chicken_id,
    x_px,
    y_px,
    confidence,
    detected,
    bbox,
    extra
)
VALUES (
    now(),
    'video_001',
    'pen_01',
    12345,
    7,
    7,
    512.4,
    348.7,
    0.96,
    True,
    '{"type":"OBB","cx":512.4,"cy":348.7,"w":80.2,"h":160.5,"angle":1.57}'::jsonb,
    '{"detector":"yolo11l-obb","tracker":"deepsort_obb"}'::jsonb
);
```

Time can also be inserted this way: `2026-06-17 10:00:00.000000-04`, `TIMESTAMPTZ '2026-06-17 10:00:00-04'+ 90 * INTERVAL '1 second' / 30`


## Query

- Average tracklet duration:
```sql
WITH tracklets AS (
    SELECT
        video_id,
        track_id,
        MIN(ts) AS start_time,
        MAX(ts) AS end_time,
        COUNT(*) AS num_points
    FROM tracking_points
    GROUP BY video_id, track_id
)
SELECT
    AVG(EXTRACT(EPOCH FROM end_time - start_time)) AS avg_tracklet_seconds,
    AVG(num_points) AS avg_points_per_tracklet
FROM tracklets;

  avg_tracklet_seconds  | avg_points_per_tracklet 
------------------------+-------------------------
 0.00000000000000000000 |  1.00000000000000000000
                      
```

<br>


- Get one chicken trajectory:

```sql
SELECT ts, frame_index, track_id, chicken_id, x_px, y_px
FROM tracking_points
WHERE video_id = 'video_001'
  AND chicken_id = 7
ORDER BY ts;

              ts               | frame_index | track_id | chicken_id | x_px  | y_px  
-------------------------------+-------------+----------+------------+-------+-------
 2026-06-17 04:02:52.755202+00 |       12345 |        7 |          7 | 512.4 | 348.7
```


- Find points inside a rectangular image region:

```sql
SELECT ts, track_id, chicken_id, x_px, y_px
FROM tracking_points
WHERE video_id = 'video_001'
  AND ST_Intersects(
        geom,
        ST_MakeEnvelope(100, 100, 600, 500, 0)
      )
ORDER BY ts;

              ts               | track_id | chicken_id | x_px  | y_px  
-------------------------------+----------+------------+-------+-------
 2026-06-17 04:02:52.755202+00 |        7 |          7 | 512.4 | 348.7
```




