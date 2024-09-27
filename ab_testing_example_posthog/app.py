from flask import Flask, render_template, make_response, request
from posthog import Posthog
import uuid
from dotenv import load_dotenv
import os

load_dotenv()

posthog = Posthog(
  os.getenv('POSTHOG_API_KEY'), 
  host='https://us.i.posthog.com'
)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/blog/<string:slug>", methods=["GET", "POST"])
def blog(slug):

  response = make_response()

  if 'user_id' not in request.cookies:
    user_id = str(uuid.uuid4())
    response.set_cookie('user_id', user_id)
  else:
    user_id = request.cookies.get('user_id')

  flag_key = "mlops-exp"
  flag = posthog.get_feature_flag(flag_key, user_id)

  if request.method == "GET":
    if (flag == 'test'):
      response.data = f"""
        <p>Welcome to the very cool blog: {slug}</p>
        <form method="post" action="/blog/{slug}">
            <input type="submit" value="Like this cool blog" name="like"/>
        </form>
      """
      return response

    response.data = f"""
      <p>Welcome to the blog post: {slug}</p>
      <form method="post" action="/blog/{slug}">
          <input type="submit" value="Like" name="like"/>
      </form>
    """
    return response


  elif request.method == "POST":
    posthog.capture(
      user_id, 
      "liked_post", 
      {
        'slug': slug,
        f'$feature/{flag_key}': flag
        
      }
    )
    return f"<p>Thanks for liking {slug}</p>"

if __name__ == '__main__':
    app.run(debug=True)