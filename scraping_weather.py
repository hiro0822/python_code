import requests
import json
import mysql.connector
import datetime


# APIキーの指定
apikey = "7243c7f4b4b8e74bdf3cbdbc1796d5de"

# 天気を調べたい都市の一覧
cities = ["Tokyo,JP","Osaka-shi,JP"]
# APIのひな型
api = "http://api.openweathermap.org/data/2.5/weather?q={city}&APPID={key}"

# 温度変換(ケルビン→摂氏)
k2c = lambda k: k - 273.15

#mysqlに接続
conn = mysql.connector.connect(user='root', password='root', host='localhost', database='weather')
cur = conn.cursor()
# 各都市の気象情報を取得する
for name in cities:
    # APIのURLを得る
    url = api.format(city=name, key=apikey)
    # 実際にAPIにリクエストを送信して結果を取得する
    r = requests.get(url)
    # 結果はJSON形式なのでデコードする
    data = json.loads(r.text)

    city_name = data["name"]
    weather = data["weather"][0]["description"]
    min_temp = k2c(data["main"]["temp_min"])
    max_temp = k2c(data["main"]["temp_max"])
    humidity= data["main"]["humidity"]
    pressure = data["main"]["pressure"]
    # if data["wind"]["deg"]!:
    #     wind_deg = nul
    # else:
    wind_deg = data["wind"]["deg"]
    wind_speed = data["wind"]["speed"]
    #時間を取得
    t = datetime.datetime.now()
    w_date = t.strftime("%Y%m%d")
    w_time = t.strftime("%H%m")
        # INSERT
    cur.execute('INSERT INTO w_info(city_name,weather,min_temp,max_temp,humidity,pressure,wind_deg,wind_speed,yyyymmdd,hhmm) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',(city_name,weather,min_temp,max_temp,humidity,pressure,wind_deg,wind_speed,w_date,w_time))
    conn.commit()
    # 結果を出力
cur.close()
conn.close()


