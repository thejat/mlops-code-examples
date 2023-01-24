 # load Flask 
import flask
import requests

app = flask.Flask(__name__)

@app.route("/", methods=["GET"])
def food():

    data = {"success": False}

    if "msg" in flask.request.args:
        data['foodname'] = str(flask.request.args['msg'])
        try:
            req = requests.get(f"https://foodish-api.herokuapp.com/api/images/{data['foodname']}")
            data["response"] = req.json()
            data["success"] = True
        except:
            pass
    else:
        try:
            req = requests.get("https://foodish-api.herokuapp.com/api/")
            data["response"] = req.json()
            data["success"] = True
        except:
            pass

    if data['success']:
        img_str= f"""
                      <img src="{data["response"]['image']}" alt="Random Food" width="500" height="600"> 
                 """
    else:
        img_str= "Food API failed"
        
    return f"""
                <!doctype html>
                <html>
                <head>
                <title>Our Funky HTML Page</title>
                <meta name="description" content="Our first page">
                <meta name="keywords" content="html tutorial template">
                <link rel="stylesheet" href="https://unpkg.com/flexgrid.io@3.0.4/dist/flexgrid.min.css" />

                </head>
                <body>
                 <div class="row xs-justify-center xs-items-center">

                    {img_str}

                 </div>
                </body>
                </html>
        """
    
# start the flask app, allow remote connections
if __name__ == '__main__':
    app.run(host='0.0.0.0')