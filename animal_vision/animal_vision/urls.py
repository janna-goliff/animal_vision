"""animal_vision URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from main.views import vision_view, camera_feed, landing, main, sources

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', landing, name='landing'),
    path('main', main, name='main'),
    path('sources', sources, name='main'),
    path('vision', vision_view, name='vision'),
    path('camera_feed/<str:animal>', camera_feed, name='camera_feed'),
]
