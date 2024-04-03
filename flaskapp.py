from spam_classifier_function import pipeline, predicter
from flask import Flask, request, render_template

app = Flask(__name__)
@app.route("/", methods = ["GET", "POST"])
def spam_checker():
    message = str(request.form.get("message"))
    result = predicter([message], pipeline)
    return "Our spam checkersays:\n"+result

if __name__ == "__main__":
	app.run(debug=True)