from fastapi import FastAPI
from joblib import load
from ML.model import predict
from starlette.responses import RedirectResponse
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi import status

app = FastAPI()

@app.get('/')
def index():
	return RedirectResponse(url='/api')

#@app.get("/items/{item_id}")
#def read_item(item_id: int, q: str = None):
#    return {"item_id": item_id, "q": q}

@app.get('/api', response_class=HTMLResponse)
def api():
	features = load("ML/utils/joblibs/features.joblib")
	features_type = load("ML/utils/joblibs/features_type.joblib")
	html = "<p>Features</p>"
	html += "<form method=\"POST\">"
	for i in range(0, len(features)):
		html += "<p>" + features[i] + "</p><input type=\""
		if features_type[i] == "float64":
			#print(features_type[i])
			html += "number"
		html += "\" name=\"" + features[i] + "\" value=\"\"><br></br>"
	html += "<input type=\"submit\" value=\"Valider\"></form>"
	return html

@app.post('/api')
async def api_post(request: Request):
	data = []
	form_data = await request.form()
	#print(form_data)
	#FormData([('sepal_length', '4'), ('sepal_width', '4'), ('petal_length', '4'), ('petal_width', '4')])
	for each in form_data:
		data.append(form_data[each])
	return RedirectResponse(url='/api/result/'+str(predict(data)), status_code=status.HTTP_303_SEE_OTHER)


@app.get('/api/result/{data}', response_class=HTMLResponse)
def result(data: str = None):
	data = data[1:-1]
	data = data.split("+")
	tmp = data
	data = []
	for each in tmp:
		if len(each) > 0:
			data.append(each)
		#each = each.rstrip()
	labels = load("ML/utils/joblibs/labels.joblib")
	html = "<h1>Label : {}</h1>".format(labels[data.index(max(data))])
	for i in range(0, len(labels)):
		html += "<p>{} : {}</p>".format(labels[i], data[i])
	return html
