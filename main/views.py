from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from main.camera import Camera

# heavily references https://github.com/sawardekar/Django_VideoStream/blob/master/streamapp/views.py

# def cameraGenerator(camera):
#     while(True):
#         frame = camera.getVideoFrame()
#         yield(b'--frame\r\n'
# 				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
# def camera_feed(request, animal):
# 	return StreamingHttpResponse(cameraGenerator(Camera(animal)),
# 					content_type='multipart/x-mixed-replace; boundary=frame')

# Create your views here.
def landing(request):
	return render(request, "landing.html", {})

def main(request):
	return render(request, "main.html", {})

def sources(request):
	return render(request, "sources.html", {})

def vision_view(request):
    isAnimal = request.GET.get("is")
    return render(request, "vision.html", {"isAnimal": isAnimal})