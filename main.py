import json
import urllib.request
from fastai.vision import *

def handler(request):
    defaults.device = torch.device('cpu')

    model = request.args['model']
    urllib.request.urlretrieve(model, '/tmp/model.pkl')
    path = Path('/tmp')
    learner = load_learner(path, 'model.pkl')
    
    image = request.args['image']
    urllib.request.urlretrieve(image, '/tmp/image.jpg')
    img = open_image('/tmp/image.jpg')
    pred_class,pred_idx,outputs = learner.predict(img)

    return json.dumps({
        "predictions": sorted(
            zip(learner.data.classes, map(float, outputs)),
            key=lambda p: p[1],
            reverse=True
        )
    })