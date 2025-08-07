import re
from datetime import datetime, timezone

ocr_text = '07-05-2025 20.05.57 Sat'

# m = re.search(r'\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2}', ocr_text)
m = re.search(
    r'(?P<mon>\d{2})[^\d]*(?P<day>\d{2})[^\d]*(?P<year>\d{4})[^\d]*(?P<hour>\d{2})[^\d]*(?P<min>\d{2})[^\d]*(?P<sec>\d{2})[^\d]*',
    ocr_text)

if m:
    timestamp_utc = m.group(0)
    print("Extracted UTC timestamp:", timestamp_utc)
    mon = int(m.group("mon"))
    day = int(m.group("day"))
    year =int( m.group("year"))
    hour =int( m.group("hour"))
    min = int(m.group("min"))
    sec = int(m.group("sec"))
    print(m.group("mon"))
    print(m.group("day"))
    print(m.group("year"))
    print(m.group("hour"))
    print(m.group("min"))
    print(m.group("sec"))

    dt = datetime(year, mon, day, hour, min, sec, tzinfo=timezone.utc)
    # get the POSIX timestamp (float, seconds since epoch)
    unix_ts = dt.timestamp()
    print(unix_ts)         # e.g. 1741679157.0
    print(int(unix_ts))    # e.g. 1741679157

else:
    print("No timestamp match found in OCR output.")
