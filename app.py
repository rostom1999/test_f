from flask import Flask
import dash_bootstrap_components
app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello rostom '


if __name__ == '__main__':
    app.run()
