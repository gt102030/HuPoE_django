from django.urls import path

from . import views

app_name = 'pybo'

urlpatterns = [
    path('', views.index, name='index'),
    path('detect', views.detect, name='detect'),
    path('nothingdetect', views.nothingdetect, name='nothingdetect'),

]