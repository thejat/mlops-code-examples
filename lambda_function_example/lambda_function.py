import json

# top_n = {'196':[(1,3),(2,4)]}
# movie_dict = {1:{'name':'a'},2:{'name':'b'}}

def lambda_handler(event,context):
data = {"success": False}


with open("top_n.json", "r") as read_file:
    top_n = json.load(read_file)
with open("movie_dict.json", "r") as read_file:
    movie_dict = json.load(read_file)


print(event) #debug
if "body" in event:
    event = event["body"]
    if event is not None:
        event = json.loads(event)
    else:
        event = {}

if "uid" in event: 
    data["response"] = str([movie_dict.get(iid,{'name':None})['name'] for (iid, _) in top_n[event.get("uid")]])
    data["success"] = True

return {
    'statusCode': 200,
    'headers':{'Content-Type':'application/json'},
    'body': json.dumps(data)
} 