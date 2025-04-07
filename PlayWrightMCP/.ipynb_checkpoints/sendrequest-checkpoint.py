import requests

url = "http://localhost:3000/mcp"
commands = [
    { "action": "goto", "url": "https://google.com" },
    { "action": "fill", "selector": "input[name='q']", "value": "chatgpt" },
    { "action": "press", "selector": "input[name='q']", "key": "Enter" }
]

res = requests.post(url, json={"commands": commands})
print(res.status_code)
print(res.json())