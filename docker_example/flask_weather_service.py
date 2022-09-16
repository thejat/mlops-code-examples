# load Flask
import flask
import requests
from flask import jsonify
from geopy.geocoders import Nominatim

app = flask.Flask(__name__)

# define a predict function as an endpoint
@app.route("/", methods=["GET", "POST"])
def weather():

    data = {"success": False}
    # https://pypi.org/project/geopy/
    geolocator = Nominatim(user_agent="cloud_function_weather_app")

    # Works with post req:
    # curl -i -H "Content-Type: application/json" -X POST -d "{\"msg\":\"Chicago\"}" localhost:5000
    # params = flask.request.json
    # if params is None:
    #     params = flask.request.args

    if flask.request.is_json:
        params = flask.request.json
    else:
        params = flask.request.args

    # params = request.get_json()
    if "msg" in params:
        location = geolocator.geocode(str(params["msg"]))
        data["location"] = [
            location.address,
            location.latitude,
            location.longitude,
            location.altitude,
        ]
        # https://www.weather.gov/documentation/services-web-api
        try:
            result1 = requests.get(
                f"https://api.weather.gov/points/{location.latitude},{location.longitude}"
            )
            result2 = requests.get(f"{result1.json()['properties']['forecast']}")
            data["response"] = result2.json()
            data["success"] = True
        except:
            pass
    return jsonify(data)


# start the flask app, allow remote connections
if __name__ == "__main__":
    app.run(host="0.0.0.0")
